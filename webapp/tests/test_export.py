"""Export YOLO : périmètre, txt vides, collisions inter-sessions, garde-fous.
L'aller-retour complet sur le dataset réel vit dans test_aller_retour.py."""
import itertools

import pytest

from conftest import STORAGE_TEST_ROOT, exec_sql, exec_sql_all, insert_image
from test_import import make_jpg

_compteur_sha = itertools.count(0x40)


def image_annotee(engine, base_ids, name, *, session="s1", batch="b1",
                  boxes=((0, 0.5, 0.5, 0.2, 0.2),), status="annotee", **cols):
    """Image au statut exportable + fichier réel + boîtes humaines validées."""
    make_jpg(STORAGE_TEST_ROOT / "p", name, "red")
    sha = f"{next(_compteur_sha):02x}" * 32
    ids = {"s1": base_ids[session], "b1": base_ids[batch]}
    if status == "relue":
        cols.setdefault("annotated_by", base_ids["alice"])
        cols.setdefault("annotated_at", "2026-07-30T10:00:00Z")
        cols.setdefault("reviewed_by", base_ids["bob"])
        cols.setdefault("reviewed_at", "2026-07-30T11:00:00Z")
    img = insert_image(engine, ids, sha, f"p/{name}", status=status, **cols)
    for cid, x, y, w, h in boxes:
        exec_sql(engine,
                 "INSERT INTO annotations (image_id, class_id, x_center,"
                 " y_center, box_width, box_height, source, state, created_by,"
                 " decided_by, decided_at) VALUES (%s, %s, %s, %s, %s, %s,"
                 " 'humain', 'validee', %s, %s, now()) RETURNING id",
                 (img, cid, x, y, w, h, base_ids["alice"], base_ids["alice"]))
    return img


def faire_export(db, sortie, session_names=None):
    from app.exporter import export_yolo
    from app.storage import get_storage

    return export_yolo(db, get_storage(), output_dir=sortie,
                       session_names=session_names)


def test_export_nominal(db, engine, base_ids, tmp_path):
    image_annotee(engine, base_ids, "avec.jpg",
                  boxes=((0, 0.5, 0.5, 0.2, 0.2), (5, 0.25, 0.25, 0.1, 0.1)))
    image_annotee(engine, base_ids, "sans.jpg", boxes=())  # sans intrus

    sortie = tmp_path / "export"
    report = faire_export(db, sortie)

    assert report.images == 2 and report.boxes == 2 and report.empty_labels == 1
    # label non vide : lignes terminées par \n, dernière comprise
    contenu = (sortie / "labels" / "avec.txt").read_text()
    assert contenu.endswith("\n") and len(contenu.splitlines()) == 2
    assert contenu.splitlines()[0].startswith("0 0.5 0.5")
    # image sans intrus : .txt VIDE (0 octet) mais PRÉSENT, image copiée
    assert (sortie / "labels" / "sans.txt").read_bytes() == b""
    assert (sortie / "images" / "sans.jpg").is_file()
    # groups.csv : group_id = ID de session (immuable), pas son nom
    lignes = (sortie / "groups.csv").read_text().splitlines()
    assert lignes[0] == "stem,group_id"
    assert set(lignes[1:]) == {f"avec,{base_ids['s1']}",
                               f"sans,{base_ids['s1']}"}
    # classes.txt : le référentiel de l'export, tracé dans la sortie
    from app.classes import class_names
    assert (sortie / "classes.txt").read_text().splitlines() == list(class_names())
    # rapport par session : id, nom, images, boîtes
    assert (base_ids["s1"], "s1", 2, 2) in report.sessions
    assert report.class_counts["Plastique"] == 1
    assert report.class_counts["Verre"] == 1
    assert report.class_counts["Eponge"] == 0


def test_proposee_et_rejetee_ne_sortent_jamais(db, engine, base_ids, tmp_path):
    img = image_annotee(engine, base_ids, "mixte.jpg",
                        boxes=((2, 0.5, 0.5, 0.2, 0.2),))
    exec_sql(engine,  # proposée non décidée
             "INSERT INTO annotations (image_id, class_id, x_center, y_center,"
             " box_width, box_height, source, state, confidence, model_name)"
             " VALUES (%s, 1, 0.4, 0.4, 0.1, 0.1, 'modele', 'proposee', 0.6,"
             " 'best.pt@aaaa') RETURNING id", (img,))
    exec_sql(engine,  # rejetée : faux positif conservé pour métriques
             "INSERT INTO annotations (image_id, class_id, x_center, y_center,"
             " box_width, box_height, source, state, confidence, model_name,"
             " decided_by, decided_at) VALUES (%s, 3, 0.6, 0.6, 0.1, 0.1,"
             " 'modele', 'rejetee', 0.7, 'best.pt@aaaa', %s, now())"
             " RETURNING id", (img, base_ids["bob"]))

    report = faire_export(db, tmp_path / "export")
    assert report.boxes == 1
    contenu = (tmp_path / "export" / "labels" / "mixte.txt").read_text()
    assert len(contenu.splitlines()) == 1 and contenu.startswith("2 ")


def test_perimetre_des_statuts(db, engine, base_ids, tmp_path):
    image_annotee(engine, base_ids, "ok1.jpg")
    image_annotee(engine, base_ids, "ok2.jpg", status="relue")
    for statut in ("en_attente_preannotation", "a_annoter", "en_cours"):
        image_annotee(engine, base_ids, f"non_{statut}.jpg", status=statut)

    report = faire_export(db, tmp_path / "export")
    assert report.images == 2
    stems = {p.stem for p in (tmp_path / "export" / "labels").iterdir()}
    assert stems == {"ok1", "ok2"}


def test_archivee_jamais_exportee(db, engine, base_ids, tmp_path):
    img = image_annotee(engine, base_ids, "arch.jpg")
    exec_sql(engine, "UPDATE images SET superseded_at = now() WHERE id = %s",
             (img,))

    report = faire_export(db, tmp_path / "export")
    assert report.images == 0
    # la session à zéro reste VISIBLE dans le rapport
    assert (base_ids["s1"], "s1", 0, 0) in report.sessions


def test_collision_inter_sessions(db, engine, base_ids, tmp_path):
    """Deux images de même nom brut issues de sessions différentes : deux
    fichiers distincts, préfixe UNIQUEMENT sur la seconde (Option B)."""
    image_annotee(engine, base_ids, "un.jpg", export_filename="IMG_0001.jpg")
    image_annotee(engine, base_ids, "deux.jpg", session="s2", batch="b2",
                  export_filename="IMG_0001.jpg",
                  boxes=((4, 0.5, 0.5, 0.2, 0.2),))

    sortie = tmp_path / "export"
    report = faire_export(db, sortie)

    stems = sorted(p.stem for p in (sortie / "labels").iterdir())
    assert stems == ["IMG_0001", "s2__IMG_0001"]
    assert report.renamed == [("IMG_0001.jpg", "s2__IMG_0001.jpg")]
    # groups.csv relie chaque stem à SA session
    lignes = set((sortie / "groups.csv").read_text().splitlines()[1:])
    assert lignes == {f"IMG_0001,{base_ids['s1']}",
                      f"s2__IMG_0001,{base_ids['s2']}"}


def test_class_id_hors_referentiel_echoue(db, engine, base_ids, tmp_path):
    from app.classes import class_count

    image_annotee(engine, base_ids, "hors.jpg",
                  boxes=((class_count(), 0.5, 0.5, 0.2, 0.2),))
    sortie = tmp_path / "export"
    with pytest.raises(ValueError, match="hors référentiel"):
        faire_export(db, sortie)
    assert not sortie.exists() and not (tmp_path / "export.part").exists()


def test_sortie_non_vide_refusee(db, engine, base_ids, tmp_path):
    image_annotee(engine, base_ids, "x.jpg")
    sortie = tmp_path / "export"
    sortie.mkdir()
    (sortie / "quelque_chose.txt").write_text("occupé")
    with pytest.raises(ValueError, match="non vide"):
        faire_export(db, sortie)
    assert (sortie / "quelque_chose.txt").read_text() == "occupé"  # intact


def test_fichier_stockage_manquant_annule_tout(db, engine, base_ids, tmp_path):
    image_annotee(engine, base_ids, "ok.jpg")
    insert_image(engine, base_ids, "e" * 64, "p/fantome.jpg", status="annotee")

    sortie = tmp_path / "export"
    with pytest.raises(ValueError, match="fichier manquant"):
        faire_export(db, sortie)
    # tout-ou-rien : même l'image valide n'a pas été écrite
    assert not sortie.exists() and not (tmp_path / "export.part").exists()


def test_session_inconnue_refusee(db, engine, base_ids, tmp_path):
    with pytest.raises(ValueError, match="inconnue"):
        faire_export(db, tmp_path / "export", session_names=["n_existe_pas"])


def test_selection_par_session(db, engine, base_ids, tmp_path):
    image_annotee(engine, base_ids, "a.jpg")
    image_annotee(engine, base_ids, "b.jpg", session="s2", batch="b2")

    report = faire_export(db, tmp_path / "export", session_names=["s2"])
    assert report.images == 1
    assert [s[0] for s in report.sessions] == [base_ids["s2"]]


def test_api_export(db, engine, base_ids, make_user, make_client, tmp_path):
    image_annotee(engine, base_ids, "api.jpg")
    make_user("chef", role="administrateur")
    make_user("annot")

    client = make_client()
    client.post("/api/auth/login",
                json={"username": "annot", "password": "motdepasse123"})
    r = client.post("/api/exports", json={"output_dir": str(tmp_path / "e1")})
    assert r.status_code == 403  # réservé à l'administrateur

    admin = make_client()
    admin.post("/api/auth/login",
               json={"username": "chef", "password": "motdepasse123"})
    r = admin.post("/api/exports", json={"output_dir": str(tmp_path / "e2")})
    assert r.status_code == 201
    corps = r.json()
    assert corps["images"] == 1 and corps["empty_labels"] == 0
    assert corps["sessions"][0]["name"] == "s1"
    assert (tmp_path / "e2" / "labels" / "api.txt").is_file()


def test_cli_export(db, engine, base_ids, tmp_path, capsys):
    from app import cli

    image_annotee(engine, base_ids, "cli.jpg")
    cli.main(["export-yolo", "--output", str(tmp_path / "sortie")])
    out = capsys.readouterr().out
    assert "1 image(s)" in out and "[id " in out
    assert (tmp_path / "sortie" / "groups.csv").is_file()
