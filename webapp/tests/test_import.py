"""Import d'un lot : périmètre lecture seule, doublons, rapport, refus propre."""
import shutil
from datetime import date

import pytest
from PIL import Image as PILImage

from conftest import STORAGE_TEST_ROOT, exec_sql, exec_sql_all
from app.importer import import_session_folder
from app.storage import get_storage


def make_jpg(directory, name, color):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    PILImage.new("RGB", (64, 48), color).save(path)
    return path


@pytest.fixture
def admin_id(make_user):
    return make_user("root", role="administrateur").id


def do_import(db, source, name="session-test", **kwargs):
    kwargs.setdefault("captured_on", date(2026, 7, 18))
    sources = source if isinstance(source, list) else [source]
    return import_session_folder(
        db, get_storage(), source_dirs=sources, name=name, **kwargs
    )


def snapshot_dir(directory):
    return {p.name: p.read_bytes() for p in sorted(directory.iterdir())}


def test_import_nominal(db, engine, tmp_path, admin_id):
    src = tmp_path / "depot"
    for i, color in enumerate(["red", "green", "blue"]):
        make_jpg(src, f"photo_{i}.jpg", color)
    avant = snapshot_dir(src)

    report = do_import(db, src, admin_id=admin_id, name="2026-07-18_tas-A",
                       lighting="naturel", camera_height_cm=120,
                       compost_state="frais", operator="Hamza")
    assert not report.aborted
    assert sorted(report.created) == ["photo_0.jpg", "photo_1.jpg", "photo_2.jpg"]
    assert report.duplicates == [] and report.rejected == []

    # la source est intacte : mêmes fichiers, mêmes octets
    assert snapshot_dir(src) == avant

    # base : session, lot « import », images en attente + événements tracés
    assert exec_sql(engine, "SELECT lighting FROM sessions WHERE id = %s",
                    (report.session_id,)) == "naturel"
    assert exec_sql(engine, "SELECT name FROM batches WHERE id = %s",
                    (report.batch_id,)) == "import"
    assert exec_sql(engine,
                    "SELECT count(*) FROM images WHERE session_id = %s"
                    " AND status = 'en_attente_preannotation'",
                    (report.session_id,)) == 3
    assert exec_sql(engine,
                    "SELECT count(*) FROM image_status_events WHERE from_status IS NULL"
                    " AND to_status = 'en_attente_preannotation'"
                    " AND changed_by = %s", (admin_id,)) == 3
    # dimensions relevées et fichiers copiés au chemin enregistré
    assert exec_sql(engine,
                    "SELECT count(*) FROM images WHERE width = 64 AND height = 48") == 3
    with engine.connect() as c:
        for rel, in c.exec_driver_sql("SELECT original_path FROM images").all():
            assert (STORAGE_TEST_ROOT / rel).is_file()


def test_doublon_dans_le_meme_dossier(db, tmp_path, admin_id):
    src = tmp_path / "depot"
    original = make_jpg(src, "a.jpg", "red")
    shutil.copyfile(original, src / "copie_de_a.jpg")

    report = do_import(db, src, admin_id=admin_id)
    assert report.created == ["a.jpg"]
    assert report.duplicates == [("copie_de_a.jpg", "identique à a.jpg (depot)")]


def test_reimport_total_refuse_sans_rien_ecrire(db, engine, tmp_path, admin_id):
    src = tmp_path / "depot"
    make_jpg(src, "a.jpg", "red")
    do_import(db, src, admin_id=admin_id, name="premiere")

    fichiers_avant = sorted(p.relative_to(STORAGE_TEST_ROOT)
                            for p in STORAGE_TEST_ROOT.rglob("*") if p.is_file())
    report = do_import(db, src, admin_id=admin_id, name="seconde")
    assert report.aborted
    assert "Rien n'a été écrit" in report.aborted_reason
    assert len(report.duplicates) == 1
    # aucune écriture : ni session, ni image, ni fichier
    assert exec_sql(engine, "SELECT count(*) FROM sessions") == 1
    assert exec_sql(engine, "SELECT count(*) FROM images") == 1
    assert sorted(p.relative_to(STORAGE_TEST_ROOT)
                  for p in STORAGE_TEST_ROOT.rglob("*") if p.is_file()) == fichiers_avant


def test_reimport_partiel_signale_et_continue(db, engine, tmp_path, admin_id):
    src1 = tmp_path / "depot1"
    deja = make_jpg(src1, "deja.jpg", "red")
    do_import(db, src1, admin_id=admin_id, name="premiere")

    src2 = tmp_path / "depot2"
    make_jpg(src2, "nouvelle.jpg", "green")
    shutil.copyfile(deja, src2 / "deja.jpg")

    report = do_import(db, src2, admin_id=admin_id, name="seconde")
    assert not report.aborted
    assert report.created == ["nouvelle.jpg"]
    assert len(report.duplicates) == 1
    assert report.duplicates[0][0] == "deja.jpg"
    assert "déjà importé (image" in report.duplicates[0][1]
    assert exec_sql(engine, "SELECT count(*) FROM images") == 2


def test_point3_sha_dont_la_seule_occurrence_est_archivee(db, engine, tmp_path,
                                                          admin_id):
    """Unicité TOTALE du sha256 : une image dont la seule occurrence en base
    est archivée (superseded_at) reste un doublon — la réimporter dans une
    autre session casserait le split par session."""
    src = tmp_path / "depot"
    fichier = make_jpg(src, "a.jpg", "red")
    first = do_import(db, src, admin_id=admin_id, name="premiere")
    exec_sql(engine, "UPDATE images SET superseded_at = now() WHERE session_id = %s",
             (first.session_id,))

    src2 = tmp_path / "depot2"
    src2.mkdir()
    shutil.copyfile(fichier, src2 / "b.jpg")

    report = do_import(db, src2, admin_id=admin_id, name="seconde")
    assert report.aborted
    assert len(report.duplicates) == 1
    assert "déjà importé" in report.duplicates[0][1]
    assert exec_sql(engine, "SELECT count(*) FROM sessions") == 1


def test_fichiers_rejetes_signales_sans_bloquer(db, tmp_path, admin_id):
    src = tmp_path / "depot"
    make_jpg(src, "valide.jpg", "red")
    (src / "notes.txt").write_text("pas une image")
    (src / "corrompue.jpg").write_bytes(b"pas du jpeg")

    report = do_import(db, src, admin_id=admin_id)
    assert report.created == ["valide.jpg"]
    assert sorted(report.rejected) == [
        ("corrompue.jpg", "fichier illisible"),
        ("notes.txt", "extension non supportée"),
    ]


def test_nom_de_session_deja_pris(db, engine, tmp_path, admin_id):
    src = tmp_path / "depot"
    make_jpg(src, "a.jpg", "red")
    do_import(db, src, admin_id=admin_id, name="unique")
    src2 = tmp_path / "depot2"
    make_jpg(src2, "b.jpg", "green")
    with pytest.raises(ValueError, match="existe déjà"):
        do_import(db, src2, admin_id=admin_id, name="unique")
    assert exec_sql(engine, "SELECT count(*) FROM sessions") == 1


def test_rattachement_a_une_session_existante(db, engine, tmp_path, admin_id):
    """Un poste importé après coup REJOINT la session existante (deux
    sessions pour la même matière recréeraient la fuite train/test) : même
    lot backlog, unicité des noms d'export préservée, session non modifiée —
    les paramètres de session sont refusés au rattachement."""
    make_jpg(tmp_path / "posteA", "photo.jpg", "red")
    make_jpg(tmp_path / "posteB", "photo.jpg", "green")  # même nom, autre contenu
    r1 = do_import(db, tmp_path / "posteA", admin_id=admin_id, name="s-ratt")
    assert not r1.aborted

    report = import_session_folder(
        db, get_storage(), source_dirs=[tmp_path / "posteB"],
        admin_id=admin_id, name="s-ratt", attach_existing=True)
    assert not report.aborted
    assert report.session_id == r1.session_id
    assert report.batch_id == r1.batch_id  # même lot backlog « import »
    # collision de nom d'export résolue par le préfixe du poste
    assert report.renamed == [("photo.jpg", "posteB__photo.jpg")]
    assert exec_sql(engine, "SELECT count(*) FROM images WHERE session_id = %s",
                    (r1.session_id,)) == 2
    assert exec_sql(engine, "SELECT count(*) FROM sessions") == 1

    # rattachement à une session inconnue : refus
    with pytest.raises(ValueError, match="introuvable"):
        import_session_folder(db, get_storage(),
                              source_dirs=[tmp_path / "posteB"],
                              admin_id=admin_id, name="s-inconnue",
                              attach_existing=True)
    # paramètres de session refusés au rattachement (do_import pose une date)
    with pytest.raises(ValueError, match="rattachement"):
        do_import(db, tmp_path / "posteB", admin_id=admin_id, name="s-ratt",
                  attach_existing=True)
    # création sans date de capture : refus
    with pytest.raises(ValueError, match="date de capture"):
        import_session_folder(db, get_storage(),
                              source_dirs=[tmp_path / "posteA"],
                              admin_id=admin_id, name="s-neuve")


def test_dossier_introuvable(db, tmp_path, admin_id):
    with pytest.raises(ValueError, match="introuvable"):
        do_import(db, tmp_path / "nexiste_pas", admin_id=admin_id)


def test_api_import_admin_et_403(make_client, make_user, tmp_path):
    make_user("root", role="administrateur", password="motdepasse123")
    make_user("alice", role="annotateur")
    src = tmp_path / "depot"
    make_jpg(src, "a.jpg", "red")
    corps = {"source_dirs": [str(src)], "name": "api-1", "captured_on": "2026-07-18"}

    alice = make_client()
    alice.post("/api/auth/login",
               json={"username": "alice", "password": "motdepasse123"})
    assert alice.post("/api/imports", json=corps).status_code == 403

    admin = make_client()
    admin.post("/api/auth/login",
               json={"username": "root", "password": "motdepasse123"})
    r = admin.post("/api/imports", json=corps)
    assert r.status_code == 201
    assert r.json()["created"] == ["a.jpg"]

    # réimport du même dossier : 409 avec le rapport en détail
    r2 = admin.post("/api/imports", json=dict(corps, name="api-2"))
    assert r2.status_code == 409
    assert len(r2.json()["detail"]["duplicates"]) == 1


def _sha_of(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_multi_dossiers_une_seule_session(db, engine, tmp_path, admin_id):
    """Trois postes qui capturent le même tas au même moment = UNE session
    (sinon le split par session fuit) ; les noms en collision sont préfixés
    du poste, les noms déjà uniques restent intacts."""
    poste_a, poste_b = tmp_path / "posteA", tmp_path / "posteB"
    make_jpg(poste_a, "IMG_0001.jpg", "red")
    make_jpg(poste_a, "IMG_0002.jpg", "green")
    make_jpg(poste_b, "IMG_0001.jpg", "blue")  # même nom, contenu différent

    report = do_import(db, [poste_a, poste_b], admin_id=admin_id)
    assert not report.aborted
    assert len(report.created) == 3
    assert exec_sql(engine, "SELECT count(*) FROM sessions") == 1
    assert report.renamed == [("IMG_0001.jpg", "posteB__IMG_0001.jpg")]
    # le nom brut n'est jamais modifié ; l'export est désambiguïsé
    rows = dict(exec_sql_all(engine,
                "SELECT export_filename, original_filename FROM images"))
    assert rows["IMG_0001.jpg"] == "IMG_0001.jpg"
    assert rows["posteB__IMG_0001.jpg"] == "IMG_0001.jpg"
    assert rows["IMG_0002.jpg"] == "IMG_0002.jpg"
    # le poste d'origine est persisté (support du découpage round-robin)
    assert dict(exec_sql_all(engine,
                "SELECT source_label, count(*) FROM images GROUP BY 1")) == {
        "posteA": 2, "posteB": 1}


def test_unicite_export_filename_en_base(engine, base_ids):
    from conftest import insert_image

    insert_image(engine, base_ids, "1" * 64, "p/x.jpg",
                 export_filename="IMG_0001.jpg")
    from conftest import refus
    refus(engine, "23505",
          "INSERT INTO images (session_id, batch_id, original_filename,"
          " export_filename, source_label, original_path, width, height, sha256)"
          " VALUES (%s, %s, 'IMG_0001.jpg', 'IMG_0001.jpg', 'p', 'p/y.jpg',"
          " 100, 100, %s)",
          (base_ids["s1"], base_ids["b1"], "2" * 64))


def test_orphelin_au_contenu_divergent_bloque(db, engine, tmp_path, admin_id):
    """Un fichier présent au chemin sha mais au contenu divergent (sauvegarde
    restaurée, copie manuelle) est une corruption : échec explicite, le
    fichier suspect n'est pas touché, rien n'est écrit."""
    src = tmp_path / "depot"
    fichier = make_jpg(src, "a.jpg", "red")
    suspect = STORAGE_TEST_ROOT / f"sessions/1/originals/{_sha_of(fichier)}.jpg"
    suspect.parent.mkdir(parents=True)
    suspect.write_bytes(b"contenu divergent")

    with pytest.raises(ValueError, match="stockage corrompu"):
        do_import(db, src, admin_id=admin_id)
    assert suspect.read_bytes() == b"contenu divergent"
    assert exec_sql(engine, "SELECT count(*) FROM sessions") == 0


def test_orphelin_du_stockage_reutilise(db, engine, tmp_path, admin_id):
    """Un fichier déjà présent au chemin sha attendu (orphelin d'un import
    interrompu) ne bloque pas l'import : contenu identique par construction,
    il est réutilisé tel quel."""
    src = tmp_path / "depot"
    fichier = make_jpg(src, "a.jpg", "red")
    # après TRUNCATE ... RESTART IDENTITY, la prochaine session a l'id 1
    orphelin = STORAGE_TEST_ROOT / f"sessions/1/originals/{_sha_of(fichier)}.jpg"
    orphelin.parent.mkdir(parents=True)
    orphelin.write_bytes(fichier.read_bytes())

    report = do_import(db, src, admin_id=admin_id)
    assert report.created == ["a.jpg"]
    assert exec_sql(engine, "SELECT count(*) FROM images") == 1


def test_rollback_ne_supprime_que_ses_propres_copies(db, engine, tmp_path,
                                                     admin_id, monkeypatch):
    """Point 2 : un échec en fin d'import ne doit jamais supprimer un fichier
    que cette exécution n'a pas créé (orphelin réutilisé, fichier partagé)."""
    src = tmp_path / "depot"
    partage = make_jpg(src, "partage.jpg", "red")
    nouveau = make_jpg(src, "nouveau.jpg", "green")
    chemin_partage = STORAGE_TEST_ROOT / f"sessions/1/originals/{_sha_of(partage)}.jpg"
    chemin_partage.parent.mkdir(parents=True)
    chemin_partage.write_bytes(partage.read_bytes())

    monkeypatch.setattr(db, "commit",
                        lambda: (_ for _ in ()).throw(RuntimeError("panne")))
    with pytest.raises(RuntimeError):
        do_import(db, src, admin_id=admin_id)

    # le fichier pré-existant est intact, la copie de cette exécution est retirée
    assert chemin_partage.exists()
    assert not (STORAGE_TEST_ROOT
                / f"sessions/1/originals/{_sha_of(nouveau)}.jpg").exists()
    monkeypatch.undo()
    assert exec_sql(engine, "SELECT count(*) FROM sessions") == 0


def test_cli_import(monkeypatch, tmp_path, engine, make_user, capsys):
    make_user("root", role="administrateur")
    src = tmp_path / "depot"
    make_jpg(src, "a.jpg", "red")

    from app import cli
    cli.main(["import-session", "--source", str(src), "--name", "cli-1",
              "--date", "2026-07-18", "--admin", "root",
              "--operator", "Ali"])
    out = capsys.readouterr().out
    assert "1 image(s) créée(s)" in out
    assert exec_sql(engine, "SELECT operator FROM sessions") == "Ali"

    # réimport : rapport affiché puis code de sortie 1
    with pytest.raises(SystemExit) as ei:
        cli.main(["import-session", "--source", str(src), "--name", "cli-2",
                  "--date", "2026-07-18", "--admin", "root"])
    assert ei.value.code == 1
    assert "IMPORT REFUSÉ" in capsys.readouterr().out
