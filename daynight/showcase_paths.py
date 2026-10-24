"""Prefer preserved inference exports; retain original training paths as fallback."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def showcase_checkpoint(name: str, fallback: str | Path) -> Path:
    packed = ROOT / "checkpoints" / "showcase" / f"{name}.pt"
    return packed if packed.is_file() else Path(fallback)
