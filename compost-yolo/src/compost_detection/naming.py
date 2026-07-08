"""Nommage des dossiers de runs : horodaté mais lisible.

Format : <prefix>_<jour>-<mois>_<heure>h<minutes>, ex. ``train_12-06_14h05``.
L'année et les secondes sont volontairement omises (lisibilité) ; en cas de
collision dans la même minute, un suffixe ``-2``, ``-3``... est ajouté.
"""

from datetime import datetime
from pathlib import Path


def run_name(prefix, when=None):
    """Ex. run_name('train') -> 'train_12-06_14h05'."""
    when = when or datetime.now()
    return f"{prefix}_{when:%d-%m_%Hh%M}"


def create_run_dir(runs_dir, prefix):
    """Crée et retourne runs_dir/<run_name> ; suffixe -2, -3... si déjà pris."""
    base = Path(runs_dir) / run_name(prefix)
    path, i = base, 2
    while path.exists():
        path = base.with_name(f"{base.name}-{i}")
        i += 1
    path.mkdir(parents=True)
    return path
