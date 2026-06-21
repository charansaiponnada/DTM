"""Tests for config.yaml parsing and structure."""
import yaml
from pathlib import Path


def _load_config():
    with open("config/config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_config_exists():
    assert Path("config/config.yaml").exists()


def test_config_has_required_sections():
    cfg = _load_config()
    assert "project" in cfg
    assert "data" in cfg
    assert "ground_classification" in cfg
    assert "dtm" in cfg
    assert "waterlogging" in cfg
    assert "drainage" in cfg


def test_config_project_crs():
    cfg = _load_config()
    assert cfg["project"]["crs"].startswith("EPSG:")


def test_config_villages_list():
    cfg = _load_config()
    villages = cfg["data"]["villages"]
    assert len(villages) >= 2
    names = [v["name"] for v in villages]
    assert "DEVDI" in names
    assert "KHAPRETA" in names


def test_config_tile_filter_naming():
    """Verify NE/NW x-ranges are correctly assigned (after fix)."""
    cfg = _load_config()
    for v in cfg["data"]["villages"]:
        if "_TILE_" not in v["name"]:
            continue
        tf = v["tile_filter"]
        name = v["name"]
        x_min, x_max, y_min, y_max = tf
        if "NE" in name or "SE" in name:
            assert x_min >= 0.5, f"{name}: NE/SE should have x right half, got {tf}"
        if "NW" in name or "SW" in name:
            assert x_max <= 0.5, f"{name}: NW/SW should have x left half, got {tf}"
        if "NE" in name or "NW" in name:
            assert y_min >= 0.5, f"{name}: NE/NW should have y top half, got {tf}"
        if "SE" in name or "SW" in name:
            assert y_max <= 0.5, f"{name}: SE/SW should have y bottom half, got {tf}"
