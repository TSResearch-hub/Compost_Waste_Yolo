"""Abstraction de stockage des fichiers images.

Une interface unique (lecture/écriture/suppression), une implémentation
système de fichiers. Les chemins manipulés sont TOUJOURS relatifs à la racine
(`STORAGE_ROOT`) — ce sont eux qui sont stockés en base. Toute écriture est
atomique (fichier temporaire puis rename) et l'écrasement est interdit :
le stockage ne réécrit jamais un fichier existant.

ATTENTION suppression : un fichier peut être référencé par PLUSIEURS lignes
`images` (une ligne de recadrage partage l'original de la ligne qu'elle
archive, nommage par sha256 oblige). Le socle ne supprime jamais rien en
dehors du rollback d'import — qui ne retire que les fichiers qu'il vient de
créer. Toute future fonctionnalité de suppression DOIT compter les références
en base (original_path ET cropped_path) avant d'appeler delete().
"""
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

from .config import get_settings


class Storage(ABC):
    @abstractmethod
    def save_file(self, relative_path: str, source: Path) -> None:
        """Copie `source` (jamais modifié ni déplacé) vers `relative_path`."""

    @abstractmethod
    def read(self, relative_path: str) -> bytes: ...

    @abstractmethod
    def exists(self, relative_path: str) -> bool: ...

    @abstractmethod
    def delete(self, relative_path: str, missing_ok: bool = False) -> None: ...


class FilesystemStorage(Storage):
    def __init__(self, root: Path):
        self.root = Path(root)

    def _resolve(self, relative_path: str) -> Path:
        target = (self.root / relative_path).resolve()
        if not target.is_relative_to(self.root.resolve()):
            raise ValueError(f"chemin hors de la racine de stockage : {relative_path}")
        return target

    def save_file(self, relative_path: str, source: Path) -> None:
        target = self._resolve(relative_path)
        if target.exists():
            raise FileExistsError(f"le stockage ne réécrit jamais : {relative_path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".part")
        shutil.copyfile(source, tmp)
        os.replace(tmp, target)  # atomique : jamais de fichier à moitié écrit

    def read(self, relative_path: str) -> bytes:
        return self._resolve(relative_path).read_bytes()

    def exists(self, relative_path: str) -> bool:
        return self._resolve(relative_path).exists()

    def delete(self, relative_path: str, missing_ok: bool = False) -> None:
        self._resolve(relative_path).unlink(missing_ok=missing_ok)


def get_storage() -> FilesystemStorage:
    return FilesystemStorage(get_settings().storage_root)
