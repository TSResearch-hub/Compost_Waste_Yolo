"""Reprise historique à session explicite : analyse seule par défaut,
exécution tout-ou-rien, labels validés contre data.yaml, absence de fichier
label = négatif (compost nu), compte dédié inactif. Commande couverture :
les motifs doivent partitionner exactement le dossier d'images."""
import pytest

from conftest import STORAGE_TEST_ROOT, exec_sql
from test_import import make_jpg


@pytest.fixture
def mixte(tmp_path):
    """Deux paires datées (téléphone) + trois photos IMG_* (smartphone), dont
    une SANS fichier label : un négatif, compost nu."""
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    labels.mkdir()
    make_jpg(images, "20260618_100000.jpg", "red")
    (labels / "20260618_100000.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    make_jpg(images, "20260618_100001.jpg", "green")
    (labels / "20260618_100001.txt").write_text("")  # txt vide : négatif aussi
    # sans retour à la ligne final : une partie du dataset réel est ainsi
    make_jpg(images, "IMG_1289.jpg", "blue")
    (labels / "IMG_1289.txt").write_text("1 0.5 0.5 0.2 0.2\n2 0.25 0.25 0.1 0.1")
    make_jpg(images, "IMG_1290.jpg", "yellow")  # PAS de label : négatif
    make_jpg(images, "IMG_1291.jpg", "purple")
    (labels / "IMG_1291.txt").write_text("7 0.5 0.5 0.1 0.1\n")
    return images, labels


def test_plan_lecture_seule(db, engine, mixte):
    from datetime import date

    from app.importer import plan_historical_session

    plan = plan_historical_session(
        db, images_dir=mixte[0], labels_dir=mixte[1], pattern="IMG_*",
        session_name="terrain_juillet", source_label="smartphone",
        captured_on=date(2026, 7, 14))
    assert plan.total_images == 3 and plan.total_boxes == 3
    assert plan.negatives == 1  # IMG_1290, sans fichier label
    assert plan.hors_motif == 2  # les datées relèvent d'autres passes
    assert plan.rejected == [] and plan.duplicates == []
    assert "création de la session « terrain_juillet »" in plan.summary()
    assert "négatifs (aucune boîte, compost nu) : 1" in plan.summary()
    # rien n'a été écrit, nulle part
    assert exec_sql(engine, "SELECT count(*) FROM sessions") == 0
    assert list(STORAGE_TEST_ROOT.iterdir()) == []


def test_creation(db, engine, make_user, make_client, mixte):
    from datetime import date

    from app.importer import (HISTORICAL_USERNAME, import_historical_session,
                              plan_historical_session)
    from app.storage import get_storage

    admin = make_user("root", role="administrateur")
    plan = plan_historical_session(
        db, images_dir=mixte[0], labels_dir=mixte[1], pattern="IMG_*",
        session_name="terrain_juillet", source_label="smartphone",
        captured_on=date(2026, 7, 14))
    report = import_historical_session(db, get_storage(), plan=plan,
                                       admin_id=admin.id)

    assert len(report.created) == 3 and report.renamed == []
    assert exec_sql(engine, "SELECT captured_on::text FROM sessions"
                    " WHERE name = 'terrain_juillet'") == "2026-07-14"
    assert exec_sql(engine, "SELECT count(*) FROM images WHERE status = 'annotee'"
                    " AND source_label = 'smartphone'"
                    " AND export_filename = original_filename") == 3
    assert exec_sql(engine, "SELECT count(*) FROM image_status_events"
                    " WHERE from_status IS NULL AND to_status = 'annotee'") == 3
    # annotations : validées, humaines, attribuées au compte dédié
    assert exec_sql(engine,
                    "SELECT count(*) FROM annotations a JOIN users u"
                    " ON u.id = a.created_by"
                    " WHERE a.state = 'validee' AND a.source = 'humain'"
                    " AND u.username = %s", (HISTORICAL_USERNAME,)) == 3
    # le négatif (IMG_1290, sans fichier label) : annotee avec zéro annotation
    assert exec_sql(engine,
                    "SELECT count(*) FROM images i WHERE i.status = 'annotee'"
                    " AND NOT EXISTS (SELECT 1 FROM annotations a"
                    " WHERE a.image_id = i.id)") == 1
    # le compte dédié est inactif : connexion impossible
    assert exec_sql(engine, "SELECT is_active FROM users WHERE username = %s",
                    (HISTORICAL_USERNAME,)) is False
    r = make_client().post("/api/auth/login", json={
        "username": HISTORICAL_USERNAME, "password": "nimporte"})
    assert r.status_code == 401


def test_rattachement(db, engine, make_user, mixte):
    """LE cas nominal : plusieurs postes ont capturé la même matière au même
    moment — leurs fichiers REJOIGNENT la même session, poste par poste (deux
    sessions pour la même matière = la fuite que le split par session doit
    empêcher). Les fichiers datés s'importent comme les autres : le nom ne
    dit plus la session."""
    from datetime import date

    from app.importer import import_historical_session, plan_historical_session
    from app.storage import get_storage

    admin = make_user("root", role="administrateur")
    plan_tel = plan_historical_session(
        db, images_dir=mixte[0], labels_dir=mixte[1], pattern="20260618_*",
        session_name="session_2026-06-18", source_label="telephone",
        captured_on=date(2026, 6, 18))
    assert plan_tel.total_images == 2 and plan_tel.negatives == 1  # txt vide
    import_historical_session(db, get_storage(), plan=plan_tel,
                              admin_id=admin.id)
    sid = exec_sql(engine, "SELECT id FROM sessions"
                   " WHERE name = 'session_2026-06-18'")

    plan = plan_historical_session(
        db, images_dir=mixte[0], labels_dir=mixte[1], pattern="IMG_*",
        session_name="session_2026-06-18", source_label="smartphone")
    assert plan.session_id == sid
    assert "rattachement à la session existante" in plan.summary()
    report = import_historical_session(db, get_storage(), plan=plan,
                                       admin_id=admin.id)
    assert report.session_id == sid

    # une seule session, un seul lot « import » réutilisé, cinq images
    assert exec_sql(engine, "SELECT count(*) FROM sessions") == 1
    assert exec_sql(engine, "SELECT count(*) FROM batches") == 1
    assert exec_sql(engine, "SELECT count(*) FROM images"
                    " WHERE session_id = %s", (sid,)) == 5
    # les deux postes cohabitent : le round-robin du découpage a de quoi mêler
    assert exec_sql(engine,
                    "SELECT count(DISTINCT source_label) FROM images") == 2

    # re-analyse : tout est désormais doublon, rien à réimporter
    plan2 = plan_historical_session(
        db, images_dir=mixte[0], labels_dir=mixte[1], pattern="IMG_*",
        session_name="session_2026-06-18", source_label="smartphone")
    assert plan2.total_images == 0 and len(plan2.duplicates) == 3


def test_cibles_incoherentes(db, base_ids, mixte):
    from datetime import date

    from app.importer import plan_historical_session

    # rattachement à une session qui n'existe pas
    with pytest.raises(ValueError, match="introuvable"):
        plan_historical_session(
            db, images_dir=mixte[0], labels_dir=mixte[1], pattern="IMG_*",
            session_name="n_existe_pas", source_label="smartphone")
    # création sous un nom déjà pris
    with pytest.raises(ValueError, match="existe déjà"):
        plan_historical_session(
            db, images_dir=mixte[0], labels_dir=mixte[1], pattern="IMG_*",
            session_name="s1", source_label="smartphone",
            captured_on=date(2026, 7, 14))


def test_rejets(db, tmp_path):
    """Illisible et label invalide restent des rejets ; l'absence de label
    n'en est plus un — c'est un négatif."""
    from datetime import date

    from app.importer import plan_historical_session

    images = tmp_path / "images"
    labels = tmp_path / "labels"
    labels.mkdir()
    make_jpg(images, "IMG_0001.jpg", "red")
    (labels / "IMG_0001.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    make_jpg(images, "IMG_0002.jpg", "green")  # sans label : négatif, PAS rejet
    (images / "IMG_0003.jpg").write_bytes(b"pas une image")
    make_jpg(images, "IMG_0004.jpg", "yellow")
    # class_id 8 hors référentiel : data.yaml n'a que 8 classes (0..7)
    (labels / "IMG_0004.txt").write_text("8 0.5 0.5 0.2 0.2\n")

    plan = plan_historical_session(
        db, images_dir=images, labels_dir=labels, pattern="IMG_*",
        session_name="terrain", source_label="smartphone",
        captured_on=date(2026, 7, 14))
    assert plan.total_images == 2 and plan.negatives == 1
    motifs = dict(plan.rejected)
    assert motifs["IMG_0003.jpg"] == "fichier illisible"
    assert "hors référentiel" in motifs["IMG_0004.jpg"]
    assert "IMG_0002.jpg" not in motifs


def test_collision_renommee(db, engine, make_user, tmp_path):
    """Deux reprises successives livrent chacune un IMG_1289.jpg différent :
    le second est renommé pour l'export, préfixé du poste — l'unicité
    (session_id, export_filename) survit au rattachement."""
    from datetime import date

    from app.importer import import_historical_session, plan_historical_session
    from app.storage import get_storage

    admin = make_user("root", role="administrateur")
    report = None
    for i, couleur in enumerate(("red", "blue")):
        images = tmp_path / f"lot{i}" / "images"
        labels = tmp_path / f"lot{i}" / "labels"
        labels.mkdir(parents=True)
        make_jpg(images, "IMG_1289.jpg", couleur)
        (labels / "IMG_1289.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        plan = plan_historical_session(
            db, images_dir=images, labels_dir=labels, pattern="IMG_*",
            session_name="terrain", source_label=f"tel{i}",
            captured_on=date(2026, 7, 14) if i == 0 else None)
        report = import_historical_session(db, get_storage(), plan=plan,
                                           admin_id=admin.id)
    assert report.renamed == [("IMG_1289.jpg", "tel1__IMG_1289.jpg")]
    assert exec_sql(engine, "SELECT count(*) FROM images") == 2


def test_cli_import_historique(make_user, mixte, engine, capsys):
    from app import cli

    make_user("root", role="administrateur")
    # --session, --poste et --motif sont obligatoires
    with pytest.raises(SystemExit):
        cli.main(["import-historique", "--images", str(mixte[0]),
                  "--labels", str(mixte[1]), "--admin", "root",
                  "--session", "terrain_juillet"])
    capsys.readouterr()

    args = ["import-historique", "--images", str(mixte[0]),
            "--labels", str(mixte[1]), "--admin", "root",
            "--session", "terrain_juillet", "--date", "2026-07-14",
            "--poste", "smartphone", "--motif", "IMG_*"]
    cli.main(args)  # analyse seule par défaut
    out = capsys.readouterr().out
    assert "ANALYSE SEULE" in out and "création de la session" in out
    assert exec_sql(engine, "SELECT count(*) FROM sessions") == 0

    cli.main(args + ["--execute"])
    out = capsys.readouterr().out
    assert "3 image(s), dont 1 négatif(s), → session « terrain_juillet »" in out
    assert exec_sql(engine, "SELECT count(*) FROM images"
                    " WHERE source_label = 'smartphone'") == 3

    # relance en rattachement (la session existe désormais, donc sans --date) :
    # tout est doublon, rien à importer
    relance = [a for a in args if a not in ("--date", "2026-07-14")]
    with pytest.raises(SystemExit) as ei:
        cli.main(relance + ["--execute"])
    assert ei.value.code == 1
    assert "Rien à importer" in capsys.readouterr().out


def test_referentiel_lu_depuis_data_yaml():
    from app.classes import class_names

    names = class_names()
    assert len(names) == 8
    assert names[0] == "Plastique" and names[5] == "Verre" and names[7] == "Eponge"


# ═══ Couverture : les motifs doivent partitionner le dossier d'images ════════


def test_couverture_partition_exacte(mixte, capsys):
    from app import cli

    cli.main(["couverture", "--images", str(mixte[0]),
              "--labels", str(mixte[1]),
              "--motif", "20260618_*", "--motif", "IMG_*"])
    out = capsys.readouterr().out
    assert "PARTITION EXACTE" in out
    assert "20260618_* : 2 fichier(s)" in out
    assert "IMG_* : 3 fichier(s)" in out


def test_couverture_en_echec(mixte, capsys):
    """Orphelin (aucun motif), recouvrement (plusieurs motifs) et label sans
    image : chacun listé, code de sortie 1 — l'oubli silencieux est le risque
    principal d'une reprise par motifs."""
    from app import cli

    (mixte[1] / "fantome.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    with pytest.raises(SystemExit) as ei:
        cli.main(["couverture", "--images", str(mixte[0]),
                  "--labels", str(mixte[1]),
                  "--motif", "20260618_*", "--motif", "IMG_1290*",
                  "--motif", "IMG_1291*"])
    assert ei.value.code == 1
    out = capsys.readouterr().out
    assert "COUVERTURE EN ÉCHEC" in out
    assert "hors de tout motif : 1" in out and "IMG_1289.jpg" in out
    assert "sous plusieurs motifs : 0" in out
    assert "labels sans image : 1" in out and "fantome.txt" in out

    with pytest.raises(SystemExit) as ei:
        cli.main(["couverture", "--images", str(mixte[0]),
                  "--labels", str(mixte[1]),
                  "--motif", "20260618_*", "--motif", "IMG_*",
                  "--motif", "*.jpg"])
    assert ei.value.code == 1
    out = capsys.readouterr().out
    assert "sous plusieurs motifs : 5" in out
    assert "IMG_1289.jpg — IMG_* + *.jpg" in out
