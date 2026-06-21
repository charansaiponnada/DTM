"""
src/preprocessing/pointnet_classifier.py
───────────────────────────────────────────
Lightweight PointNet-based ground classifier for LiDAR point clouds.

Architecture
------------
A simplified PointNet (Qi et al., 2017) for binary segmentation
(ground vs non-ground). Uses shared MLP → max-pool → per-point scores.

This module provides:
  1. A PyTorch model definition (PointNetGroundBinary)
  2. Training helper with geometric feature augmentation
  3. Inference helper that writes classified LAS

Reference
---------
Qi, C. R., et al. (2017). PointNet: Deep Learning on Point Sets
for 3D Classification and Segmentation. CVPR.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from loguru import logger

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    logger.warning("PyTorch not available. PointNet classifier disabled.")
    torch = None


# ══════════════════════════════════════════════════════════════════════════
#  Model Definition
# ══════════════════════════════════════════════════════════════════════════


class PointNetGroundBinary(nn.Module):
    """
    Lightweight PointNet for binary ground/non-ground classification.

    Operates on (x, y, z) + optional geometric features.
    Uses shared MLP → max-pool global feature → per-point classification.

    Parameters
    ----------
    in_channels : number of input features (3 for xyz, or more with extra feats)
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.in_channels = in_channels

        # Shared MLP (point-wise)
        self.mlp1 = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        # Per-point classification head
        self.head = nn.Sequential(
            nn.Linear(256 + 256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2),  # logits: non-ground, ground
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, N, C)  batch of point clouds

        Returns
        -------
        scores : (B, N, 2)  per-point class logits
        """
        B, N, C = x.shape
        # Shared MLP: (B, N, 256)
        point_feats = self.mlp1(x.reshape(B * N, C)).reshape(B, N, 256)

        # Max-pool global feature: (B, 256)
        global_feat, _ = point_feats.max(dim=1)

        # Concatenate per-point + global: (B, N, 512)
        global_feat_expanded = global_feat.unsqueeze(1).expand(-1, N, -1)
        combined = torch.cat([point_feats, global_feat_expanded], dim=-1)

        # Head: (B, N, 2)
        scores = self.head(combined.reshape(B * N, 512)).reshape(B, N, 2)
        return scores


# ══════════════════════════════════════════════════════════════════════════
#  Training
# ══════════════════════════════════════════════════════════════════════════


def train_pointnet(
    xyz: np.ndarray,
    labels: np.ndarray,
    val_split: float = 0.2,
    batch_size: int = 8192,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    model_save_path: Optional[str | Path] = None,
) -> PointNetGroundBinary:
    """
    Train PointNet on subsampled point cloud patches.

    Parameters
    ----------
    xyz    : (N, 3) point coordinates
    labels : (N,) binary labels (1=ground, 0=non-ground)
    """
    if torch is None:
        raise ImportError("PyTorch required for PointNet training")

    N = len(xyz)
    logger.info(f"Training PointNet on {N:,} points …")

    # Subsample for balanced training
    rng = np.random.default_rng(42)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    n_train = min(100_000, min(len(pos_idx), len(neg_idx)) * 2)
    half = n_train // 2
    if len(pos_idx) < half or len(neg_idx) < half:
        half = min(len(pos_idx), len(neg_idx))
        n_train = half * 2
    idx = np.concatenate([
        rng.choice(pos_idx, size=half, replace=False),
        rng.choice(neg_idx, size=half, replace=False),
    ])
    rng.shuffle(idx)

    xyz_sampled = xyz[idx].astype(np.float32)
    labels_sampled = labels[idx].astype(np.int64)

    # Normalise coordinates to unit sphere
    centroid = xyz_sampled.mean(axis=0)
    scale = np.max(np.linalg.norm(xyz_sampled - centroid, axis=1))
    xyz_norm = (xyz_sampled - centroid) / (scale + 1e-8)

    # Train/val split
    n_val = int(len(xyz_norm) * val_split)
    xyz_train, xyz_val = xyz_norm[:-n_val], xyz_norm[-n_val:]
    y_train, y_val = labels_sampled[:-n_val], labels_sampled[-n_val:]

    # Normalise per-batch: create random subsets within each epoch
    model = PointNetGroundBinary(in_channels=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = 0.0
    for epoch in range(epochs):
        model.train()
        # Shuffle and create random mini-batches
        perm = torch.randperm(len(xyz_train))
        losses = []
        for start in range(0, len(perm), batch_size):
            batch_idx = perm[start:start + batch_size]
            bx = torch.from_numpy(xyz_train[batch_idx.numpy()]).unsqueeze(0).to(device)
            by = torch.from_numpy(y_train[batch_idx.numpy()]).unsqueeze(0).to(device)

            optimizer.zero_grad()
            logits = model(bx)
            loss = F.cross_entropy(logits.reshape(-1, 2), by.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(torch.from_numpy(xyz_val).unsqueeze(0).to(device))
            val_preds = val_logits.argmax(dim=-1).cpu().numpy().ravel()
            val_acc = (val_preds == y_val).mean()
            tp = ((val_preds == 1) & (y_val == 1)).sum()
            fp = ((val_preds == 1) & (y_val == 0)).sum()
            fn = ((val_preds == 0) & (y_val == 1)).sum()
            val_f1 = 2 * tp / (2 * tp + fp + fn + 1e-10)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), str(model_save_path)) if model_save_path else None

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"  Epoch {epoch+1:2d}/{epochs}  loss={np.mean(losses):.4f}  "
                f"val_acc={val_acc:.4f}  val_F1={val_f1:.4f}"
            )

    logger.success(f"PointNet training complete. Best val F1 = {best_val_f1:.4f}")
    if model_save_path:
        logger.info(f"Model saved → {model_save_path}")
    return model


# ══════════════════════════════════════════════════════════════════════════
#  Inference
# ══════════════════════════════════════════════════════════════════════════


def apply_pointnet(
    las_input: str | Path | "laspy.LasData",
    model: PointNetGroundBinary | str | Path,
    batch_size: int = 200_000,
    device: str = "cpu",
    ground_class_code: int = 2,
) -> "laspy.LasData":
    """
    Apply trained PointNet to classify an entire LAS file.

    Parameters
    ----------
    las_input   : LAS file path or laspy.LasData object
    model       : trained PointNetGroundBinary or path to state_dict
    batch_size  : points per inference batch (memory control)

    Returns
    -------
    laspy.LasData with updated classification field
    """
    if torch is None:
        raise ImportError("PyTorch required for PointNet inference")

    import laspy

    if isinstance(las_input, (str, Path)):
        las = laspy.read(str(las_input))
    else:
        las = las_input

    if isinstance(model, (str, Path)):
        state_dict = torch.load(str(model), map_location=device)
        in_channels = state_dict["mlp1.0.weight"].shape[1]
        pn = PointNetGroundBinary(in_channels=in_channels).to(device)
        pn.load_state_dict(state_dict)
        model = pn

    model.eval()
    xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float32)
    N = len(xyz)

    # Normalise using training stats
    centroid = xyz.mean(axis=0)
    scale = np.max(np.linalg.norm(xyz - centroid, axis=1))
    xyz_norm = (xyz - centroid) / (scale + 1e-8)

    all_preds = []
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = torch.from_numpy(xyz_norm[start:end]).unsqueeze(0).to(device)
            logits = model(batch)
            preds = logits.argmax(dim=-1).cpu().numpy().ravel()
            all_preds.append(preds)

    predictions = np.concatenate(all_preds)
    cls = np.where(predictions == 1, ground_class_code, 0).astype(np.uint8)
    las.classification = cls

    n_ground = int((cls == ground_class_code).sum())
    logger.info(f"PointNet: {n_ground:,} / {N:,} points → ground ({100*n_ground/N:.1f}%)")
    return las


# ══════════════════════════════════════════════════════════════════════════
#  Convenience: Full PointNet Pipeline
# ══════════════════════════════════════════════════════════════════════════


def classify_with_pointnet(
    input_path: str | Path,
    output_path: str | Path,
    smrf_labels_path: Optional[str | Path] = None,
    model_path: Optional[str | Path] = None,
    train_epochs: int = 40,
    device: str = "cpu",
) -> Path:
    """
    Train (if needed) and apply PointNet ground classification.

    If no model_path is given, trains from scratch using SMRF labels
    as training signal.

    Returns
    -------
    Path to classified LAS
    """
    if torch is None:
        raise ImportError("PyTorch required for PointNet")

    import laspy

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    las = laspy.read(str(input_path))
    xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)

    if smrf_labels_path:
        smrf_las = laspy.read(str(smf_labels_path))
        labels = (np.array(smrf_las.classification) == 2).astype(int)
    else:
        logger.warning("No SMRF labels provided — using z-percentile heuristic for PointNet training")
        z = np.array(las.z, dtype=np.float32)
        z_low = np.percentile(z, 5)
        labels = (z <= z_low + 0.5).astype(int)

    model = None
    if model_path and Path(model_path).exists():
        logger.info(f"Loading pre-trained PointNet: {model_path}")
        state_dict = torch.load(str(model_path), map_location=device)
        in_channels = state_dict["mlp1.0.weight"].shape[1]
        model = PointNetGroundBinary(in_channels=in_channels).to(device)
        model.load_state_dict(state_dict)
    else:
        logger.info("Training PointNet from scratch …")
        model = train_pointnet(
            xyz, labels,
            epochs=train_epochs,
            device=device,
            model_save_path=model_path,
        )

    las = apply_pointnet(las, model, device=device)
    las.write(str(output_path))
    logger.success(f"PointNet classification complete → {output_path}")
    return output_path
