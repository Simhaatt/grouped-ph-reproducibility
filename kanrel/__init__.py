"""Compatibility namespace for the original experiment imports."""
from pathlib import Path
__path__ = [str(Path(__file__).resolve().parent.parent / "src" / "kanrel")]
