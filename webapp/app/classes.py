"""Référentiel de classes — lu depuis data.yaml (DATA_YAML_PATH), unique
source de vérité. Aucune liste recopiée en dur : l'ordre du fichier définit
les index, et il ne doit jamais changer par insertion au milieu.
"""
from functools import lru_cache
from pathlib import Path

import yaml

from .config import get_settings


def lire_data_yaml(path: Path) -> tuple[str, ...]:
    """Lit et valide un data.yaml : liste `names` non vide de chaînes, dans
    l'ordre du fichier. ValueError sinon (yaml.YAMLError si illisible). Sans
    cache — sert aussi à contrôler un fichier téléversé
    (POST /api/models_ia/upload) avant de l'accepter."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    names = data.get("names") if isinstance(data, dict) else None
    if not isinstance(names, list) or not names or not all(
        isinstance(n, str) and n for n in names
    ):
        raise ValueError(f"data.yaml invalide ({path}) : liste `names` attendue")
    return tuple(names)


@lru_cache
def _load(path: str) -> tuple[str, ...]:
    return lire_data_yaml(Path(path))


def class_names() -> tuple[str, ...]:
    """Noms de classes, dans l'ordre du data.yaml (index = id YOLO)."""
    return _load(str(get_settings().data_yaml_path))


def class_count() -> int:
    return len(class_names())
