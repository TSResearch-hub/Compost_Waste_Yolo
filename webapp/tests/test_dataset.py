"""Écran Dataset (administrateur) : export YOLO en ZIP à la volée — même
périmètre que l'export vers un répertoire, vérifications avant le premier
octet — et dépôt d'images brutes depuis le PC dans la session
« Import_Manuel_Admin », créée puis rejointe."""
import io
import tempfile
import zipfile
from datetime import date
from pathlib import Path

import pytest

from conftest import STORAGE_TEST_ROOT, exec_sql, exec_sql_all
from test_admin_technique import login
from test_export import image_annotee
from test_sync import make_zip


@pytest.fixture
def comptes(make_user):
    make_user("root", role="administrateur")
    make_user("annotatrice")


def exporter(client):
    return client.get("/api/dataset/export")


def deposer(client, zip_path):
    with open(zip_path, "rb") as fh:
        return client.post("/api/dataset/import",
                           files={"archive": ("photos.zip", fh, "application/zip")})


def dossiers_temporaires():
    return sorted(p for p in Path(tempfile.gettempdir()).iterdir()
                  if p.name.startswith("compost_dataset_"))


# ── Accès ────────────────────────────────────────────────────────────────────

def test_administrateur_seulement(make_client, comptes, tmp_path):
    z = make_zip(tmp_path / "a.zip", {"a.jpg": "red"})
    anonyme, annotatrice = make_client(), login(make_client(), "annotatrice")
    assert anonyme.get("/api/dataset/resume").status_code == 401
    assert exporter(anonyme).status_code == 401
    assert deposer(anonyme, z).status_code == 401
    assert annotatrice.get("/api/dataset/resume").status_code == 403
    assert exporter(annotatrice).status_code == 403
    assert deposer(annotatrice, z).status_code == 403


# ── Export ───────────────────────────────────────────────────────────────────

def test_export_zip_nominal(make_client, comptes, engine, base_ids):
    from app.classes import class_names
    from app.config import get_settings

    image_annotee(engine, base_ids, "avec.jpg",
                  boxes=((0, 0.5, 0.5, 0.2, 0.2), (5, 0.25, 0.25, 0.1, 0.1)))
    image_annotee(engine, base_ids, "sans.jpg", boxes=())      # négatif
    image_annotee(engine, base_ids, "encours.jpg", status="en_cours")  # hors périmètre
    root = login(make_client(), "root")

    # le résumé annonce exactement ce que l'archive contiendra
    resume = root.get("/api/dataset/resume")
    assert resume.status_code == 200, resume.text
    assert (resume.json()["images"], resume.json()["boxes"],
            resume.json()["empty_labels"]) == (2, 2, 1)
    assert resume.json()["class_counts"]["Plastique"] == 1
    assert resume.json()["class_counts"]["Verre"] == 1

    r = exporter(root)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/zip"
    assert r.headers["content-disposition"].startswith('attachment; filename="dataset_yolo_')
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    assert zf.testzip() is None
    assert set(zf.namelist()) == {
        "images/avec.jpg", "labels/avec.txt", "images/sans.jpg", "labels/sans.txt",
        "data.yaml", "groups.csv", "classes.txt", "rapport.txt"}
    # images : les octets du stockage, sans recompression (JPEG déjà compressé)
    assert zf.read("images/avec.jpg") == (STORAGE_TEST_ROOT / "p" / "avec.jpg").read_bytes()
    assert zf.getinfo("images/avec.jpg").compress_type == zipfile.ZIP_STORED
    # labels : mêmes règles que l'export vers un répertoire
    label = zf.read("labels/avec.txt").decode()
    assert label.endswith("\n") and len(label.splitlines()) == 2
    assert label.splitlines()[0].startswith("0 0.5 0.5")
    assert zf.read("labels/sans.txt") == b""
    # data.yaml : le référentiel du serveur, tel quel
    assert zf.read("data.yaml") == Path(get_settings().data_yaml_path).read_bytes()
    assert zf.read("classes.txt").decode().splitlines() == list(class_names())
    lignes = zf.read("groups.csv").decode().splitlines()
    assert lignes[0] == "stem,group_id"
    assert set(lignes[1:]) == {f"avec,{base_ids['s1']}", f"sans,{base_ids['s1']}"}
    rapport = zf.read("rapport.txt").decode()
    assert "2 image(s), 2 boîte(s)" in rapport
    assert "images ignorées, fichier absent du stockage : 0" in rapport
    assert resume.json()["fichiers_manquants"] == 0
    # rien n'a été écrit côté serveur
    assert not any(p.name.endswith(".part") for p in STORAGE_TEST_ROOT.rglob("*"))


def test_export_fichier_absent_ignore_et_compte(make_client, comptes, engine,
                                                base_ids):
    """Un fichier absent du stockage n'empêche plus le téléchargement : l'image
    est écartée, l'archive est complète pour le reste et rapport.txt le dit."""
    image_annotee(engine, base_ids, "ok.jpg")
    image_annotee(engine, base_ids, "fantome.jpg")
    (STORAGE_TEST_ROOT / "p" / "fantome.jpg").unlink()
    root = login(make_client(), "root")

    resume = root.get("/api/dataset/resume").json()
    assert resume["images"] == 1 and resume["fichiers_manquants"] == 1

    r = exporter(root)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/zip"
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    assert zf.testzip() is None
    assert set(zf.namelist()) == {"images/ok.jpg", "labels/ok.txt", "data.yaml",
                                  "groups.csv", "classes.txt", "rapport.txt"}
    rapport = zf.read("rapport.txt").decode()
    assert "1 image(s), 1 boîte(s)" in rapport
    assert "images ignorées, fichier absent du stockage : 1" in rapport
    assert zf.read("groups.csv").decode().splitlines()[1:] == [f"ok,{base_ids['s1']}"]


def test_export_class_id_hors_referentiel_refuse(make_client, comptes, engine,
                                                 base_ids):
    from app.classes import class_count

    image_annotee(engine, base_ids, "hors.jpg",
                  boxes=((class_count(), 0.5, 0.5, 0.2, 0.2),))
    root = login(make_client(), "root")
    assert exporter(root).status_code == 400
    r = root.get("/api/dataset/resume")
    assert r.status_code == 400 and "hors référentiel" in r.json()["detail"]


def test_resume_base_vierge(make_client, comptes):
    r = login(make_client(), "root").get("/api/dataset/resume")
    assert r.status_code == 200
    assert r.json() == {"images": 0, "boxes": 0, "empty_labels": 0,
                        "sessions": [], "class_counts": {}, "renamed": [],
                        "fichiers_manquants": 0}


# ── Import ───────────────────────────────────────────────────────────────────

def test_import_cree_la_session_puis_la_rejoint(make_client, comptes, engine,
                                                tmp_path):
    from app.importer import SESSION_IMPORT_MANUEL

    root = login(make_client(), "root")
    avant = dossiers_temporaires()

    r = deposer(root, make_zip(tmp_path / "un.zip",
                               {"a.jpg": "red", "b.jpg": "green"},
                               extra={"notes.txt": b"pas une image"}))
    assert r.status_code == 201, r.text
    rapport = r.json()
    assert rapport["session_name"] == SESSION_IMPORT_MANUEL
    assert sorted(rapport["created"]) == ["a.jpg", "b.jpg"]
    assert rapport["rejected"] == [["notes.txt", "extension non supportée"]]
    session_id = rapport["session_id"]
    # session créée aujourd'hui, par l'administrateur connecté, lot « import »
    assert exec_sql_all(
        engine, "SELECT name, captured_on, jetson_id FROM sessions") == [
        (SESSION_IMPORT_MANUEL, date.today(), None)]
    assert exec_sql(engine, "SELECT username FROM users u JOIN sessions s"
                            " ON s.created_by = u.id") == "root"
    assert exec_sql_all(engine, "SELECT name FROM batches WHERE session_id = %s",
                        (session_id,)) == [("import",)]
    # les images tombent dans la file de pré-annotation, fichiers en place
    assert exec_sql_all(
        engine, "SELECT status, source_label, original_path FROM images"
                " ORDER BY original_filename") == [
        ("en_attente_preannotation", "root", f"sessions/{session_id}/originals/" +
         exec_sql(engine, "SELECT sha256 FROM images WHERE original_filename = 'a.jpg'")
         + ".jpg"),
        ("en_attente_preannotation", "root", f"sessions/{session_id}/originals/" +
         exec_sql(engine, "SELECT sha256 FROM images WHERE original_filename = 'b.jpg'")
         + ".jpg"),
    ]
    for rel, in exec_sql_all(engine, "SELECT original_path FROM images"):
        assert (STORAGE_TEST_ROOT / rel).is_file()

    # second dépôt : la session est REJOINTE, pas dupliquée ; un doublon du
    # premier dépôt est ignoré sans faire échouer le reste
    r = deposer(root, make_zip(tmp_path / "deux.zip", {"a.jpg": "red", "c.jpg": "blue"}))
    assert r.status_code == 201, r.text
    assert r.json()["session_id"] == session_id
    assert r.json()["created"] == ["c.jpg"]
    assert len(r.json()["duplicates"]) == 1 and r.json()["duplicates"][0][0] == "a.jpg"
    assert exec_sql(engine, "SELECT count(*) FROM sessions") == 1
    assert exec_sql(engine, "SELECT count(*) FROM images") == 3
    # dossiers temporaires nettoyés
    assert dossiers_temporaires() == avant


def test_import_tout_en_doublon_409_avec_rapport(make_client, comptes, engine,
                                                 tmp_path):
    root = login(make_client(), "root")
    z = make_zip(tmp_path / "a.zip", {"a.jpg": "red"})
    assert deposer(root, z).status_code == 201
    r = deposer(root, z)
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["aborted_reason"] and detail["created"] == []
    assert detail["duplicates"][0][0] == "a.jpg"
    assert exec_sql(engine, "SELECT count(*) FROM images") == 1


def test_import_archive_illisible_ou_trop_grosse(make_client, comptes, engine,
                                                 tmp_path, monkeypatch):
    from app.config import get_settings

    root = login(make_client(), "root")
    avant = dossiers_temporaires()
    pas_un_zip = tmp_path / "faux.zip"
    pas_un_zip.write_bytes(b"ceci n'est pas une archive")
    r = deposer(root, pas_un_zip)
    assert r.status_code == 400 and "illisible" in r.json()["detail"]
    # entrée qui tente de sortir du dossier d'extraction
    with zipfile.ZipFile(tmp_path / "slip.zip", "w") as zf:
        zf.writestr("../evasion.jpg", b"x")
    r = deposer(root, tmp_path / "slip.zip")
    assert r.status_code == 400 and "interdit" in r.json()["detail"]
    # plafond de taille (le même que les envois des cartes)
    monkeypatch.setenv("SYNC_MAX_UPLOAD_MB", "0")
    get_settings.cache_clear()
    try:
        r = deposer(root, make_zip(tmp_path / "gros.zip", {"a.jpg": "red"}))
        assert r.status_code == 413
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()
    # rien n'a été écrit, dossiers temporaires nettoyés
    assert exec_sql(engine, "SELECT count(*) FROM sessions") == 0
    assert exec_sql(engine, "SELECT count(*) FROM images") == 0
    assert dossiers_temporaires() == avant
