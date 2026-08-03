"""Abstraction de stockage système de fichiers."""
import pytest

from app.storage import FilesystemStorage


@pytest.fixture
def storage(tmp_path):
    return FilesystemStorage(tmp_path / "racine")


def test_cycle_ecrire_lire_supprimer(storage, tmp_path):
    src = tmp_path / "photo.jpg"
    src.write_bytes(b"contenu")
    storage.save_file("sessions/1/originals/abc.jpg", src)
    assert storage.exists("sessions/1/originals/abc.jpg")
    assert storage.read("sessions/1/originals/abc.jpg") == b"contenu"
    # la source n'a été ni déplacée ni modifiée
    assert src.read_bytes() == b"contenu"
    storage.delete("sessions/1/originals/abc.jpg")
    assert not storage.exists("sessions/1/originals/abc.jpg")


def test_jamais_d_ecrasement(storage, tmp_path):
    src = tmp_path / "a.jpg"
    src.write_bytes(b"v1")
    storage.save_file("x.jpg", src)
    src.write_bytes(b"v2")
    with pytest.raises(FileExistsError):
        storage.save_file("x.jpg", src)
    assert storage.read("x.jpg") == b"v1"


def test_chemin_hors_racine_refuse(storage, tmp_path):
    src = tmp_path / "a.jpg"
    src.write_bytes(b"x")
    with pytest.raises(ValueError):
        storage.save_file("../evasion.jpg", src)
    with pytest.raises(ValueError):
        storage.read("../../etc/passwd")


def test_delete_missing_ok(storage):
    with pytest.raises(FileNotFoundError):
        storage.delete("absent.jpg")
    storage.delete("absent.jpg", missing_ok=True)
