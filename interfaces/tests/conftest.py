"""Rend les modules de l'interface d'annotation importables depuis les tests."""
import sys
from pathlib import Path

ANNOTATION_DIR = Path(__file__).resolve().parent.parent / "annotation"
if str(ANNOTATION_DIR) not in sys.path:
    sys.path.insert(0, str(ANNOTATION_DIR))
