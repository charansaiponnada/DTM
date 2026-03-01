"""
src/ml/pointnet_classifier.py
────────────────────────────────
Lightweight PointNet implementation for point cloud ground classification.

Architecture based on:
  Qi et al., "PointNet: Deep Learning on Point Sets for 3D Classification
  and Segmentation", CVPR 2017.

Adaptations for MoPR hackathon:
  - 3 input features (X, Y, Z) + 3 geometric features = 6 channels
  - Binary classification: ground (1) vs non-ground (0)
  - Input normalization handles zero-intensity datasets (Gujarat data)
  - Tiled inference for memory efficiency with 64M–163M point clouds

Usage
-----
  model = PointNetClassifier(n_features=6)
  trainer = PointNetTrainer(model, device="cuda")
  trainer.fit(xyz_array, labels)
  preds = trainer.predict(xyz_array)
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from loguru import logger
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════════
#  Dataset
# ══════════════════════════════════════════════════════════════════════════

class PointCloudDataset(Dataset):
    """
    Patch-based point cloud dataset.
    Each sample is a random local neighbourhood (patch) of n_points points.
    """

    def __init__(
        self,
        xyz: np.ndarray,           # (N, 3) or (N, F)
        labels: np.ndarray,        # (N,) int
        patch_size: int = 1024,
        n_patches: int  = 5000,
        augment: bool   = True,
    ):
        super().__init__()
        self.xyz       = torch.tensor(xyz,    dtype=torch.float32)
        self.labels    = torch.tensor(labels, dtype=torch.long)
        self.patch_size = patch_size
        self.n_patches  = n_patches
        self.augment    = augment
        self.N          = len(xyz)

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx: int):
        # Random anchor point
        anchor = np.random.randint(self.N)
        start  = max(0, anchor - self.patch_size // 2)
        end    = min(self.N, start + self.patch_size)
        idxs   = np.arange(start, end)

        # Pad if too small
        if len(idxs) < self.patch_size:
            pad = np.random.choice(idxs, self.patch_size - len(idxs))
            idxs = np.concatenate([idxs, pad])

        pts = self.xyz[idxs]        # (P, F)
        lbl = self.labels[idxs]     # (P,)
        idxs_t = torch.tensor(idxs, dtype=torch.long)  # (P,) point indices

        # Centre the patch (critical for PointNet input invariance)
        pts = pts - pts.mean(dim=0, keepdim=True)
        pts_max = pts.abs().max()
        if pts_max > 0:
            pts = pts / pts_max    # scale to unit sphere

        # Data augmentation
        if self.augment:
            # Random jitter
            pts += torch.randn_like(pts) * 0.005
            # Random rotation around Z axis
            theta = np.random.uniform(0, 2 * np.pi)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            R = torch.tensor([[cos_t, -sin_t, 0],
                               [sin_t,  cos_t, 0],
                               [0,      0,     1]], dtype=torch.float32)
            pts[:, :3] = pts[:, :3] @ R.T

        return pts, lbl, idxs_t


# ══════════════════════════════════════════════════════════════════════════
#  PointNet Architecture
# ══════════════════════════════════════════════════════════════════════════

class TNet(nn.Module):
    """Mini T-Net for input/feature space alignment (joint alignment network)."""

    def __init__(self, k: int = 3):
        super().__init__()
        self.k  = k
        self.fc = nn.Sequential(
            nn.Linear(k, 64),   nn.ReLU(), nn.BatchNorm1d(64),
            nn.Linear(64, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, k * k),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, P, k)
        B, P, _ = x.shape
        feat = x.max(dim=1)[0]         # (B, k)  global max
        mat  = self.fc(feat)            # (B, k*k)
        mat  = mat.view(B, self.k, self.k)
        eye  = torch.eye(self.k, device=x.device).unsqueeze(0).expand(B, -1, -1)
        return mat + eye               # residual around identity


class PointNetSegmentation(nn.Module):
    """
    PointNet for per-point segmentation (binary ground/non-ground).

    Input  : (B, P, F)  – batch of point patches with F features
    Output : (B, P, 2)  – per-point class logits
    """

    def __init__(self, n_features: int = 6, n_classes: int = 2):
        super().__init__()
        self.n_features = n_features
        self.tnet3      = TNet(k=3)     # align raw XYZ

        # Shared MLP on points (implemented as Conv1d for speed)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(n_features, 64, 1),  nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 1),         nn.ReLU(), nn.BatchNorm1d(128),
        )
        self.tnet_feat = TNet(k=128)    # align feature space

        self.mlp2 = nn.Sequential(
            nn.Conv1d(128, 256, 1), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Conv1d(256, 512, 1), nn.ReLU(), nn.BatchNorm1d(512),
        )

        # Segmentation head: concatenate local (128) + global (512) features
        self.seg_head = nn.Sequential(
            nn.Conv1d(512 + 128, 256, 1), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Conv1d(256, 128, 1),       nn.ReLU(), nn.BatchNorm1d(128),
            nn.Conv1d(128, n_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, P, F)
        B, P, F = x.shape

        # Align XYZ
        T3   = self.tnet3(x[:, :, :3])
        xyz  = x[:, :, :3] @ T3             # (B, P, 3)
        if F > 3:
            x = torch.cat([xyz, x[:, :, 3:]], dim=-1)  # keep extra features
        else:
            x = xyz

        # Local features
        x_t = x.transpose(1, 2)             # (B, F, P) for Conv1d
        local_feat = self.mlp1(x_t)          # (B, 128, P)

        # Feature alignment
        local_t   = local_feat.transpose(1, 2)  # (B, P, 128)
        T_feat    = self.tnet_feat(local_t)
        local_t   = local_t @ T_feat
        local_feat = local_t.transpose(1, 2)    # (B, 128, P)

        # Global features via max pooling
        global_feat = self.mlp2(local_feat)      # (B, 512, P)
        global_max  = global_feat.max(dim=2, keepdim=True)[0]  # (B, 512, 1)
        global_exp  = global_max.expand(-1, -1, P)               # (B, 512, P)

        # Concatenate local + global
        combined = torch.cat([local_feat, global_exp], dim=1)   # (B, 640, P)
        logits   = self.seg_head(combined)                        # (B, 2, P)
        return logits.transpose(1, 2)                             # (B, P, 2)


# ══════════════════════════════════════════════════════════════════════════
#  Trainer
# ══════════════════════════════════════════════════════════════════════════

class PointNetTrainer:
    """Handles training, validation, and inference for PointNetSegmentation."""

    def __init__(
        self,
        model: Optional[PointNetSegmentation] = None,
        n_features: int = 6,
        device: str = "auto",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        logger.info(f"PointNet device: {self.device}")

        self.model = model or PointNetSegmentation(n_features=n_features)
        self.model = self.model.to(self.device)

        self.optim    = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optim, step_size=20, gamma=0.5
        )

    def fit(
        self,
        xyz: np.ndarray,           # (N, F)
        labels: np.ndarray,        # (N,)  0/1
        epochs: int        = 50,
        batch_size: int    = 16,
        patch_size: int    = 1024,
        val_size: float    = 0.1,
        save_path: Optional[str | Path] = None,
    ) -> "PointNetTrainer":

        # Class weights for imbalanced datasets (ground points often fewer)
        n_pos = labels.sum()
        n_neg = len(labels) - n_pos
        pos_w = torch.tensor([1.0, n_neg / (n_pos + 1e-6)], device=self.device)
        criterion = nn.CrossEntropyLoss(weight=pos_w)

        # Train/val split
        idx_tr, idx_val = train_test_split(
            np.arange(len(xyz)), test_size=val_size, stratify=labels, random_state=42
        )
        train_ds = PointCloudDataset(xyz[idx_tr], labels[idx_tr], patch_size)
        val_ds   = PointCloudDataset(xyz[idx_val], labels[idx_val], patch_size, augment=False)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
        val_dl   = DataLoader(val_ds,   batch_size=batch_size, num_workers=2)

        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            # ── Train ────────────────────────────────────────────────────
            self.model.train()
            tr_loss = 0.0
            for pts, lbl, _ in tqdm(train_dl, desc=f"Epoch {epoch}/{epochs}", leave=False):
                pts, lbl = pts.to(self.device), lbl.to(self.device)
                self.optim.zero_grad()
                logits = self.model(pts)               # (B, P, 2)
                loss   = criterion(logits.view(-1, 2), lbl.view(-1))
                loss.backward()
                self.optim.step()
                tr_loss += loss.item()

            # ── Validate ─────────────────────────────────────────────────
            self.model.eval()
            val_loss, val_acc = 0.0, 0.0
            with torch.no_grad():
                for pts, lbl, _ in val_dl:
                    pts, lbl = pts.to(self.device), lbl.to(self.device)
                    logits   = self.model(pts)
                    loss     = criterion(logits.view(-1, 2), lbl.view(-1))
                    val_loss += loss.item()
                    preds     = logits.argmax(dim=-1)
                    val_acc  += (preds == lbl).float().mean().item()

            tr_loss  /= len(train_dl)
            val_loss /= len(val_dl)
            val_acc  /= len(val_dl)
            self.scheduler.step()

            logger.info(
                f"Epoch {epoch:3d}  train_loss={tr_loss:.4f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            )

            if val_loss < best_val_loss and save_path:
                best_val_loss = val_loss
                self.save(save_path)
                logger.info(f"  ↳ Best model saved (val_loss={val_loss:.4f})")

        return self

    def predict(
        self,
        xyz: np.ndarray,
        batch_size: int = 32,
        patch_size: int = 1024,
    ) -> np.ndarray:
        """
        Predict per-point labels for a full point cloud array.
        Uses a sliding window approach for large inputs.

        Returns
        -------
        predictions : (N,) int array  (0=non-ground, 1=ground)
        """
        self.model.eval()
        N = len(xyz)
        votes  = np.zeros((N, 2), dtype=np.float64)   # accumulate softmax probability votes
        counts = np.zeros(N,      dtype=np.int32)      # number of patch votes per point

        # Use enough patches to cover every point multiple times
        n_patches = max(N // 50 + 100, 200)
        dummy_labels = np.zeros(N, dtype=np.int32)
        ds = PointCloudDataset(
            xyz, dummy_labels, patch_size, n_patches=n_patches, augment=False
        )
        dl = DataLoader(ds, batch_size=batch_size, num_workers=0)

        with torch.no_grad():
            for pts, _, patch_idxs in tqdm(dl, desc="Inference"):
                pts = pts.to(self.device)
                logits = self.model(pts)                     # (B, P, 2)
                probs  = F.softmax(logits, dim=-1).cpu().numpy()  # (B, P, 2)
                idxs_np = patch_idxs.numpy()                # (B, P) – original point indices

                # Accumulate per-point softmax votes
                B, P, _ = probs.shape
                for b in range(B):
                    np.add.at(votes,  idxs_np[b], probs[b])   # (P, 2) added at indices
                    np.add.at(counts, idxs_np[b], 1)

        # Points with zero votes (unsampled edge cases) default to non-ground
        unvisited = counts == 0
        if unvisited.any():
            logger.warning(
                f"{unvisited.sum()} points received no patch votes – defaulting to class 0"
            )
            counts[unvisited] = 1   # avoid division by zero

        ground_prob = votes[:, 1] / counts
        return (ground_prob > 0.5).astype(np.int32)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "n_features":  self.model.n_features,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "PointNetTrainer":
        data = torch.load(path, map_location="cpu")
        obj  = cls(n_features=data["n_features"])
        obj.model.load_state_dict(data["model_state"])
        obj.model.eval()
        return obj
