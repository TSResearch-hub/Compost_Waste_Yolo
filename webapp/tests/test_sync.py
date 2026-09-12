"""Réception des envois automatiques des cartes Jetson (POST /api/sync/upload) :
jeton partagé, carte déclarée, ZIP décompressé à plat puis importé comme un
poste de capture, session du jour créée puis rejointe, nettoyage du dossier
temporaire dans tous les cas."""
import io
import os
import tempfile
import zipfile
from datetime import date
from pathlib import Path

import pytest
from PIL import Image as PILImage

from conftest import STORAGE_TEST_ROOT, exec_sql, exec_sql_all

JETSON = "48-b0-2d-3e-aa-01"
TOKEN = "jeton-de-test-suffisamment-long"


@pytest.fixture
def sync_env(monkeypatch):
    """SYNC_TOKEN posé pour le test ; la config (lru_cache) est rechargée à
    l'entrée et après restauration de l'environnement."""
    from app.config import get_settings

    monkeypatch.setenv("SYNC_TOKEN", TOKEN)
    get_settings.cache_clear()
    yield monkeypatch
    monkeypatch.undo()
    get_settings.cache_clear()


@pytest.fixture
def carte(db):
    from app.models import JetsonDevice

    db.add(JetsonDevice(id=JETSON, name="Jetson tas A"))
    db.commit()
    return JETSON


def make_zip(path, images, extra=None, compression=zipfile.ZIP_DEFLATED):
    """ZIP de JPEG unis (même couleur = mêmes octets = doublon) + entrées
    brutes optionnelles {nom: octets}."""
    with zipfile.ZipFile(path, "w", compression) as zf:
        for name, color in images.items():
            buf = io.BytesIO()
            PILImage.new("RGB", (64, 48), color).save(buf, format="JPEG")
            zf.writestr(name, buf.getvalue())
        for name, data in (extra or {}).items():
            zf.writestr(name, data)
    return path


def upload(client, zip_path, *, jetson=JETSON, token=TOKEN, via="headers",
           **form):
    headers, data = {}, {k: v for k, v in form.items() if v is not None}
    if via == "headers":
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        if jetson is not None:
            headers["X-Jetson-Id"] = jetson
    else:
        if token is not None:
            data["token"] = token
        if jetson is not None:
            data["jetson_id"] = jetson
    with open(zip_path, "rb") as fh:
        return client.post(
            "/api/sync/upload", headers=headers, data=data,
            files={"archive": ("envoi.zip", fh, "application/zip")})


def dossiers_temporaires():
    return sorted(p for p in Path(tempfile.gettempdir()).iterdir()
                  if p.name.startswith("compost_sync_"))


def test_reception_desactivee_sans_jeton_configure(client, carte, tmp_path,
                                                   monkeypatch):
    from app.config import get_settings

    monkeypatch.delenv("SYNC_TOKEN", raising=False)
    get_settings.cache_clear()
    try:
        z = make_zip(tmp_path / "a.zip", {"a.jpg": "red"})
        r = upload(client, z)
        assert r.status_code == 503
        assert "SYNC_TOKEN" in r.json()["detail"]
    finally:
        get_settings.cache_clear()


def test_jeton_requis_et_verifie(client, sync_env, carte, tmp_path, engine):
    z = make_zip(tmp_path / "a.zip", {"a.jpg": "red"})
    assert upload(client, z, token=None).status_code == 401
    assert upload(client, z, token="mauvais").status_code == 401
    assert upload(client, z, token="mauvais", via="form").status_code == 401
    # un refus de jeton n'écrit rien — ni session, ni compte système
    assert exec_sql(engine, "SELECT count(*) FROM sessions") == 0
    assert exec_sql(engine, "SELECT count(*) FROM users") == 0
    # jeton accepté en en-tête Bearer comme en champ de formulaire
    assert upload(client, z, via="form", captured_on="2026-09-12").status_code == 201
    z2 = make_zip(tmp_path / "b.zip", {"b.jpg": "green"})
    assert upload(client, z2, captured_on="2026-09-12").status_code == 201


def test_carte_inconnue_desactivee_ou_absente(client, sync_env, carte, tmp_path,
                                              engine):
    z = make_zip(tmp_path / "a.zip", {"a.jpg": "red"})
    assert upload(client, z, jetson=None).status_code == 422
    assert upload(client, z, jetson="tas/A").status_code == 422
    r = upload(client, z, jetson="00-00-00-00-00-99")
    assert r.status_code == 403 and "inconnue" in r.json()["detail"]
    exec_sql(engine, "UPDATE jetson_devices SET is_active = false WHERE id = %s",
             (JETSON,))
    r = upload(client, z)
    assert r.status_code == 403 and "désactivée" in r.json()["detail"]
    assert exec_sql(engine, "SELECT count(*) FROM sessions") == 0


def test_envoi_nominal_puis_rattachement_du_jour(client, sync_env, carte,
                                                 tmp_path, engine):
    avant = dossiers_temporaires()
    z = make_zip(tmp_path / "a.zip", {"IMG_0001.jpg": "red",
                                      "IMG_0002.jpg": "green"})
    r = upload(client, z, captured_on="2026-09-12")
    assert r.status_code == 201, r.text
    rapport = r.json()
    assert sorted(rapport["created"]) == ["IMG_0001.jpg", "IMG_0002.jpg"]
    assert rapport["duplicates"] == [] and rapport["rejected"] == []
    assert rapport["session_name"] == f"{JETSON}_2026-09-12"
    session_id = rapport["session_id"]

    # session : nom par défaut, date envoyée, origine = la carte
    nom, capture, jetson_id, created_by = exec_sql_all(
        engine, "SELECT name, captured_on, jetson_id, created_by FROM sessions"
        " WHERE id = %s", (session_id,))[0]
    assert (nom, str(capture), jetson_id) == (
        f"{JETSON}_2026-09-12", "2026-09-12", JETSON)
    # compte système inactif, sans créateur, porteur de toute la traçabilité
    username, actif, role, cree_par = exec_sql_all(
        engine, "SELECT username, is_active, role, created_by FROM users"
        " WHERE id = %s", (created_by,))[0]
    assert (username, actif, role, cree_par) == (
        "sync_jetson", False, "annotateur", None)
    assert exec_sql(engine, "SELECT created_by FROM batches WHERE id = %s",
                    (rapport["batch_id"],)) == created_by
    assert exec_sql(engine, "SELECT name FROM batches WHERE id = %s",
                    (rapport["batch_id"],)) == "import"
    assert exec_sql(engine,
                    "SELECT count(*) FROM image_status_events WHERE from_status"
                    " IS NULL AND to_status = 'en_attente_preannotation'"
                    " AND changed_by = %s", (created_by,)) == 2
    # images : poste = identifiant de la carte, en attente, fichiers copiés
    lignes = exec_sql_all(
        engine, "SELECT source_label, status, original_path FROM images"
        " WHERE session_id = %s", (session_id,))
    assert len(lignes) == 2
    for label, statut, rel in lignes:
        assert (label, statut) == (JETSON, "en_attente_preannotation")
        assert (STORAGE_TEST_ROOT / rel).is_file()

    # second envoi du même jour (MAC brute, majuscules et `:`) : la session
    # est REJOINTE — une nouvelle image, un doublon ignoré, rien de plus
    z2 = make_zip(tmp_path / "b.zip", {"IMG_0002.jpg": "green",
                                       "IMG_0003.jpg": "blue"})
    r = upload(client, z2, jetson="48:B0:2D:3E:AA:01", captured_on="2026-09-12")
    assert r.status_code == 201, r.text
    rapport = r.json()
    assert rapport["session_id"] == session_id
    assert rapport["created"] == ["IMG_0003.jpg"]
    assert len(rapport["duplicates"]) == 1
    assert rapport["duplicates"][0][0] == "IMG_0002.jpg"
    assert exec_sql(engine, "SELECT count(*) FROM sessions") == 1
    assert exec_sql(engine, "SELECT count(*) FROM images") == 3
    assert exec_sql(engine, "SELECT count(*) FROM users") == 1  # pas recréé
    # le dossier temporaire ne survit pas à la requête
    assert dossiers_temporaires() == avant


def test_sans_date_la_session_est_celle_du_jour(client, sync_env, carte,
                                                tmp_path, engine):
    z = make_zip(tmp_path / "a.zip", {"a.jpg": "red"})
    r = upload(client, z)
    assert r.status_code == 201, r.text
    assert r.json()["session_name"] == f"{JETSON}_{date.today().isoformat()}"
    assert str(exec_sql(engine, "SELECT captured_on FROM sessions")) == (
        date.today().isoformat())


def test_session_explicite_rejoint_une_session_manuelle(client, sync_env, carte,
                                                        tmp_path, engine, db,
                                                        make_user):
    """Un nom de session fourni rattache l'envoi à une session existante,
    même manuelle : des postes qui photographient la même matière doivent
    partager une session (split train/test). La session n'est pas modifiée,
    son origine reste celle de l'import manuel (jetson_id NULL)."""
    from app.models import CaptureSession

    admin = make_user("root", role="administrateur")
    db.add(CaptureSession(name="tas-A", captured_on=date(2026, 9, 1),
                          operator="Hamza", created_by=admin.id))
    db.commit()
    z = make_zip(tmp_path / "a.zip", {"a.jpg": "red"})
    r = upload(client, z, session_name="tas-A", notes="ignorées en rattachement")
    assert r.status_code == 201, r.text
    assert r.json()["session_name"] == "tas-A"
    assert exec_sql(engine, "SELECT count(*) FROM sessions") == 1
    assert exec_sql_all(
        engine, "SELECT jetson_id, operator, notes FROM sessions")[0] == (
        None, "Hamza", None)
    assert exec_sql(engine, "SELECT source_label FROM images") == JETSON
    # et un nom de session NOUVEAU crée la session, qui porte la carte
    z2 = make_zip(tmp_path / "b.zip", {"b.jpg": "green"})
    r = upload(client, z2, session_name="tas-B", captured_on="2026-09-02",
               notes="nuit")
    assert r.status_code == 201, r.text
    assert exec_sql_all(
        engine, "SELECT jetson_id, notes FROM sessions WHERE name = 'tas-B'"
    )[0] == (JETSON, "nuit")


def test_archive_imbriquee_a_plat_et_metadonnees_ignorees(client, sync_env,
                                                          carte, tmp_path,
                                                          engine):
    z = make_zip(tmp_path / "a.zip",
                 {"b.jpg": "red", "sub/a.jpg": "green", "sub/deep/c.png": "blue"},
                 extra={"__MACOSX/._b.jpg": b"meta", ".hidden.jpg": b"x",
                        "notes.txt": b"bonjour", "dossier/": b""})
    r = upload(client, z, captured_on="2026-09-12")
    assert r.status_code == 201, r.text
    rapport = r.json()
    assert sorted(rapport["created"]) == ["b.jpg", "sub__a.jpg",
                                          "sub__deep__c.png"]
    assert rapport["rejected"] == [["notes.txt", "extension non supportée"]]
    assert sorted(n for n, in exec_sql_all(
        engine, "SELECT original_filename FROM images")) == [
        "b.jpg", "sub__a.jpg", "sub__deep__c.png"]


def test_archive_invalide_ou_zip_slip(client, sync_env, carte, tmp_path, engine):
    avant = dossiers_temporaires()
    pas_un_zip = tmp_path / "a.zip"
    pas_un_zip.write_bytes(b"ceci n'est pas une archive")
    r = upload(client, pas_un_zip, captured_on="2026-09-12")
    assert r.status_code == 400 and "illisible" in r.json()["detail"]

    z = make_zip(tmp_path / "b.zip", {"a.jpg": "red", "../evasion.jpg": "green"})
    r = upload(client, z, captured_on="2026-09-12")
    assert r.status_code == 400 and "chemin interdit" in r.json()["detail"]
    assert exec_sql(engine, "SELECT count(*) FROM sessions") == 0
    assert exec_sql(engine, "SELECT count(*) FROM images") == 0
    assert list(STORAGE_TEST_ROOT.rglob("*")) == []
    assert dossiers_temporaires() == avant


def test_rien_d_importable_409_avec_rapport(client, sync_env, carte, tmp_path,
                                            engine):
    z = make_zip(tmp_path / "a.zip", {"a.jpg": "red"})
    assert upload(client, z, captured_on="2026-09-12").status_code == 201
    # ré-envoi intégral après coupure réseau : tout en doublon, rien écrit
    r = upload(client, z, captured_on="2026-09-12")
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert "Rien n'a été écrit" in detail["aborted_reason"]
    assert len(detail["duplicates"]) == 1
    assert exec_sql(engine, "SELECT count(*) FROM images") == 1
    # archive vide : rien non plus — et pas de session ouverte pour rien
    vide = make_zip(tmp_path / "vide.zip", {})
    r = upload(client, vide, captured_on="2026-09-13")
    assert r.status_code == 409
    assert exec_sql(engine, "SELECT count(*) FROM sessions") == 1


def test_plafonds_de_taille_413(client, sync_env, carte, tmp_path, engine):
    from app.config import get_settings

    gros = os.urandom(1_600_000)  # incompressible, > 1 Mo
    sync_env.setenv("SYNC_MAX_UNZIPPED_MB", "1")
    get_settings.cache_clear()
    z = make_zip(tmp_path / "a.zip", {"a.jpg": "red"}, extra={"gros.bin": gros})
    r = upload(client, z, captured_on="2026-09-12")
    assert r.status_code == 413 and "SYNC_MAX_UNZIPPED_MB" in r.json()["detail"]

    sync_env.setenv("SYNC_MAX_UNZIPPED_MB", "2048")
    sync_env.setenv("SYNC_MAX_UPLOAD_MB", "1")
    get_settings.cache_clear()
    z = make_zip(tmp_path / "b.zip", {"a.jpg": "red"}, extra={"gros.bin": gros},
                 compression=zipfile.ZIP_STORED)
    r = upload(client, z, captured_on="2026-09-12")
    assert r.status_code == 413 and "SYNC_MAX_UPLOAD_MB" in r.json()["detail"]
    assert exec_sql(engine, "SELECT count(*) FROM sessions") == 0
    assert dossiers_temporaires() == []
