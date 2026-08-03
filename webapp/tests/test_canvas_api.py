"""API du canvas (segment 1) : droits par verrou de lot, transitions de
statut tracées, enregistrement à état complet (propositions toutes tranchées,
rejetées conservées, négatif valide), navigation, avancement."""
import itertools
from datetime import date, datetime, timezone

import pytest

from conftest import STORAGE_TEST_ROOT, exec_sql, insert_image
from test_import import make_jpg

_sha = itertools.count(0x20)


def poser_image(engine, ids, name, status="a_annoter", **cols):
    """Ligne images + fichier réel dans le stockage de test."""
    make_jpg(STORAGE_TEST_ROOT / "canvas", name, "red")
    sha = f"{next(_sha):02x}" * 32
    return insert_image(engine, ids, sha, f"canvas/{name}", status=status,
                        **cols)


def proposer(engine, image_id, class_id, x, y, w, h, confiance=0.42):
    """Boîte proposée par le modèle (state='proposee', sans décideur)."""
    return exec_sql(
        engine,
        "INSERT INTO annotations (image_id, class_id, x_center, y_center,"
        " box_width, box_height, source, state, confidence, model_name)"
        " VALUES (%s, %s, %s, %s, %s, %s, 'modele', 'proposee', %s,"
        " 'best.pt@test') RETURNING id",
        (image_id, class_id, x, y, w, h, confiance),
    )


def login(client, username, password="motdepasse123"):
    r = client.post("/api/auth/login",
                    json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return client


@pytest.fixture
def ctx(db, engine, make_user):
    """Un lot verrouillé par alice : une image pré-annotée (2 propositions),
    une image vierge, une image encore en file de pré-annotation."""
    from app.models import Batch, CaptureSession

    admin = make_user("root", role="administrateur")
    alice = make_user("alice")
    bob = make_user("bob")
    session = CaptureSession(name="s1", captured_on=date(2026, 7, 1),
                             created_by=admin.id)
    db.add(session)
    db.flush()
    batch = Batch(session_id=session.id, name="import", created_by=admin.id)
    db.add(batch)
    db.commit()
    ids = {"s1": session.id, "b1": batch.id}
    exec_sql(engine,
             "INSERT INTO batch_assignments (batch_id, user_id, assigned_by)"
             " VALUES (%s, %s, %s) RETURNING id",
             (batch.id, alice.id, admin.id))
    i_prop = poser_image(engine, ids, "prop.jpg")
    p1 = proposer(engine, i_prop, 0, 0.30, 0.30, 0.10, 0.10)
    p2 = proposer(engine, i_prop, 1, 0.60, 0.60, 0.20, 0.20, confiance=0.17)
    i_vierge = poser_image(engine, ids, "vierge.jpg")
    i_file = poser_image(engine, ids, "file.jpg",
                         status="en_attente_preannotation")
    return {"admin": admin, "alice": alice, "bob": bob, **ids,
            "i_prop": i_prop, "p1": p1, "p2": p2,
            "i_vierge": i_vierge, "i_file": i_file}


def test_ouvrir_droits_transition_et_contenu(make_client, ctx, engine):
    # non authentifié
    assert make_client().post(
        f"/api/images/{ctx['i_prop']}/ouvrir").status_code == 401
    # bob ne détient pas le lot
    assert login(make_client(), "bob").post(
        f"/api/images/{ctx['i_prop']}/ouvrir").status_code == 403

    alice = login(make_client(), "alice")
    r = alice.post(f"/api/images/{ctx['i_prop']}/ouvrir")
    assert r.status_code == 200
    corps = r.json()
    assert corps["statut"] == "en_cours" and corps["nom"] == "prop.jpg"
    assert corps["largeur"] == 100 and corps["hauteur"] == 100  # cols en base
    assert corps["classes"][0] == "Plastique" and len(corps["classes"]) == 8
    assert [(b["id"], b["state"], b["source"]) for b in corps["boites"]] == [
        (ctx["p1"], "proposee", "modele"), (ctx["p2"], "proposee", "modele")]
    assert corps["boites"][0]["confidence"] == 0.42
    # transition tracée, une seule fois même en rouvrant
    alice.post(f"/api/images/{ctx['i_prop']}/ouvrir")
    assert exec_sql(engine, "SELECT count(*) FROM image_status_events"
                    " WHERE image_id = %s AND from_status = 'a_annoter'"
                    " AND to_status = 'en_cours'", (ctx["i_prop"],)) == 1
    # l'admin ouvre sans verrou
    assert login(make_client(), "root").post(
        f"/api/images/{ctx['i_vierge']}/ouvrir").status_code == 200
    # encore en file de pré-annotation : pas ouvrable
    assert alice.post(
        f"/api/images/{ctx['i_file']}/ouvrir").status_code == 409
    # image archivée : introuvable
    exec_sql(engine, "UPDATE images SET superseded_at = %s WHERE id = %s"
             " RETURNING id", (datetime.now(timezone.utc), ctx["i_vierge"]))
    assert alice.post(
        f"/api/images/{ctx['i_vierge']}/ouvrir").status_code == 404


def test_fichier_authentifie_et_crop_prioritaire(make_client, ctx, engine):
    from app.storage import get_storage

    assert make_client().get(
        f"/api/images/{ctx['i_prop']}/fichier").status_code == 401
    assert login(make_client(), "bob").get(
        f"/api/images/{ctx['i_prop']}/fichier").status_code == 403

    alice = login(make_client(), "alice")
    r = alice.get(f"/api/images/{ctx['i_prop']}/fichier")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.content == get_storage().read("canvas/prop.jpg")

    # le crop prime : c'est à lui que les coordonnées se rapportent
    make_jpg(STORAGE_TEST_ROOT / "crops", "c.jpg", "blue")
    ids = {"s1": ctx["s1"], "b1": ctx["b1"]}
    i_crop = poser_image(engine, ids, "entier.jpg", cropped_path="crops/c.jpg",
                         crop_x=0, crop_y=0, cropped_width=40,
                         cropped_height=30)
    r = alice.get(f"/api/images/{i_crop}/fichier")
    assert r.content == get_storage().read("crops/c.jpg")
    # et l'ouverture annonce les dimensions du crop, pas de l'original
    corps = alice.post(f"/api/images/{i_crop}/ouvrir").json()
    assert (corps["largeur"], corps["hauteur"]) == (40, 30)


def test_enregistrer_nominal(make_client, ctx, engine):
    alice = login(make_client(), "alice")
    alice.post(f"/api/images/{ctx['i_prop']}/ouvrir")
    r = alice.put(f"/api/images/{ctx['i_prop']}/annotations", json={"boites": [
        # p1 validée, géométrie retouchée et classe corrigée
        {"id": ctx["p1"], "class_id": 2, "x_center": 0.35, "y_center": 0.32,
         "box_width": 0.12, "box_height": 0.11, "etat": "validee"},
        # p2 rejetée — les coordonnées envoyées seront IGNORÉES :
        # le faux positif du modèle est conservé tel qu'il l'a produit
        {"id": ctx["p2"], "class_id": 1, "x_center": 0.9, "y_center": 0.9,
         "box_width": 0.05, "box_height": 0.05, "etat": "rejetee"},
        # nouvelle boîte humaine (Eponge, jamais proposée par le modèle)
        {"class_id": 7, "x_center": 0.5, "y_center": 0.5,
         "box_width": 0.2, "box_height": 0.2, "etat": "validee"},
    ]})
    assert r.status_code == 200, r.text
    assert r.json() == {"statut": "annotee", "validees": 2, "rejetees": 1,
                        "supprimees": 0}

    # p1 : validée, retouchée, décidée par alice
    assert exec_sql(engine, "SELECT state || '/' || class_id::text"
                    " FROM annotations WHERE id = %s", (ctx["p1"],)) == "validee/2"
    assert exec_sql(engine, "SELECT x_center FROM annotations WHERE id = %s",
                    (ctx["p1"],)) == 0.35
    assert exec_sql(engine, "SELECT updated_by FROM annotations WHERE id = %s",
                    (ctx["p1"],)) == ctx["alice"].id
    # p2 : rejetée, géométrie du modèle intacte
    assert exec_sql(engine, "SELECT state FROM annotations WHERE id = %s",
                    (ctx["p2"],)) == "rejetee"
    assert exec_sql(engine, "SELECT x_center FROM annotations WHERE id = %s",
                    (ctx["p2"],)) == 0.60
    # nouvelle : humaine validée, auteur alice
    assert exec_sql(engine, "SELECT count(*) FROM annotations"
                    " WHERE image_id = %s AND source = 'humain'"
                    " AND state = 'validee' AND class_id = 7"
                    " AND created_by = %s",
                    (ctx["i_prop"], ctx["alice"].id)) == 1
    # plus aucune proposition en suspens ; image annotée, événement tracé
    assert exec_sql(engine, "SELECT count(*) FROM annotations"
                    " WHERE image_id = %s AND state = 'proposee'",
                    (ctx["i_prop"],)) == 0
    assert exec_sql(engine, "SELECT status || '/' || annotated_by::text"
                    " FROM images WHERE id = %s",
                    (ctx["i_prop"],)) == f"annotee/{ctx['alice'].id}"
    assert exec_sql(engine, "SELECT count(*) FROM image_status_events"
                    " WHERE image_id = %s AND from_status = 'en_cours'"
                    " AND to_status = 'annotee'", (ctx["i_prop"],)) == 1


def test_enregistrer_gardes(make_client, ctx, engine):
    alice = login(make_client(), "alice")
    alice.post(f"/api/images/{ctx['i_prop']}/ouvrir")
    valide_p1 = {"id": ctx["p1"], "class_id": 0, "x_center": 0.3,
                 "y_center": 0.3, "box_width": 0.1, "box_height": 0.1,
                 "etat": "validee"}

    # proposition absente du corps : aucune décision implicite
    r = alice.put(f"/api/images/{ctx['i_prop']}/annotations",
                  json={"boites": [valide_p1]})
    assert r.status_code == 422 and "sans décision" in r.json()["detail"]
    # rien n'a bougé
    assert exec_sql(engine, "SELECT state FROM annotations WHERE id = %s",
                    (ctx["p1"],)) == "proposee"
    assert exec_sql(engine, "SELECT status FROM images WHERE id = %s",
                    (ctx["i_prop"],)) == "en_cours"

    rejette_p2 = {"id": ctx["p2"], "class_id": 1, "x_center": 0.6,
                  "y_center": 0.6, "box_width": 0.2, "box_height": 0.2,
                  "etat": "rejetee"}
    # class_id hors data.yaml, vérifié côté serveur
    hors = dict(valide_p1, class_id=8)
    assert alice.put(f"/api/images/{ctx['i_prop']}/annotations",
                     json={"boites": [hors, rejette_p2]}).status_code == 422
    # boîte d'une autre image
    autre = dict(valide_p1, id=999999)
    assert alice.put(f"/api/images/{ctx['i_prop']}/annotations",
                     json={"boites": [autre, rejette_p2]}).status_code == 422
    # une boîte nouvelle ne peut pas naître rejetée
    neuve_rejetee = {"class_id": 0, "x_center": 0.5, "y_center": 0.5,
                     "box_width": 0.1, "box_height": 0.1, "etat": "rejetee"}
    assert alice.put(
        f"/api/images/{ctx['i_prop']}/annotations",
        json={"boites": [valide_p1, rejette_p2, neuve_rejetee]},
    ).status_code == 422
    # image jamais ouverte : pas de transition a_annoter -> annotee
    assert alice.put(f"/api/images/{ctx['i_vierge']}/annotations",
                     json={"boites": []}).status_code == 409
    # lot non détenu
    assert login(make_client(), "bob").put(
        f"/api/images/{ctx['i_prop']}/annotations",
        json={"boites": [valide_p1, rejette_p2]}).status_code == 403


def test_negatif_valide(make_client, ctx, engine):
    """Tout rejeter est un enregistrement valide : l'image est un négatif."""
    alice = login(make_client(), "alice")
    alice.post(f"/api/images/{ctx['i_prop']}/ouvrir")
    r = alice.put(f"/api/images/{ctx['i_prop']}/annotations", json={"boites": [
        {"id": ctx["p1"], "class_id": 0, "x_center": 0.3, "y_center": 0.3,
         "box_width": 0.1, "box_height": 0.1, "etat": "rejetee"},
        {"id": ctx["p2"], "class_id": 1, "x_center": 0.6, "y_center": 0.6,
         "box_width": 0.2, "box_height": 0.2, "etat": "rejetee"},
    ]})
    assert r.status_code == 200
    assert r.json() == {"statut": "annotee", "validees": 0, "rejetees": 2,
                        "supprimees": 0}
    # les faux positifs du modèle sont conservés, aucune boîte validée
    assert exec_sql(engine, "SELECT count(*) FROM annotations"
                    " WHERE image_id = %s AND state = 'rejetee'",
                    (ctx["i_prop"],)) == 2
    assert exec_sql(engine, "SELECT count(*) FROM annotations"
                    " WHERE image_id = %s AND state = 'validee'",
                    (ctx["i_prop"],)) == 0
    assert exec_sql(engine, "SELECT status FROM images WHERE id = %s",
                    (ctx["i_prop"],)) == "annotee"


def test_reedition(make_client, ctx, engine):
    """Rouvrir une image annotée : la boîte du modèle retirée du canvas
    redevient rejetée (jamais supprimée), la boîte humaine retirée est
    supprimée."""
    alice = login(make_client(), "alice")
    alice.post(f"/api/images/{ctx['i_prop']}/ouvrir")
    alice.put(f"/api/images/{ctx['i_prop']}/annotations", json={"boites": [
        {"id": ctx["p1"], "class_id": 0, "x_center": 0.3, "y_center": 0.3,
         "box_width": 0.1, "box_height": 0.1, "etat": "validee"},
        {"id": ctx["p2"], "class_id": 1, "x_center": 0.6, "y_center": 0.6,
         "box_width": 0.2, "box_height": 0.2, "etat": "rejetee"},
        {"class_id": 3, "x_center": 0.5, "y_center": 0.5,
         "box_width": 0.2, "box_height": 0.2, "etat": "validee"},
    ]})

    r = alice.post(f"/api/images/{ctx['i_prop']}/ouvrir")
    assert r.status_code == 200 and r.json()["statut"] == "en_cours"
    assert exec_sql(engine, "SELECT count(*) FROM image_status_events"
                    " WHERE image_id = %s AND from_status = 'annotee'"
                    " AND to_status = 'en_cours'", (ctx["i_prop"],)) == 1
    # état complet SANS p1 (validée) ni la boîte humaine ; p2 reste absente
    # aussi : déjà rejetée, elle le reste sans réapparaître dans le corps
    r = alice.put(f"/api/images/{ctx['i_prop']}/annotations",
                  json={"boites": []})
    assert r.status_code == 200
    assert r.json() == {"statut": "annotee", "validees": 0, "rejetees": 1,
                        "supprimees": 1}
    assert exec_sql(engine, "SELECT state FROM annotations WHERE id = %s",
                    (ctx["p1"],)) == "rejetee"
    assert exec_sql(engine, "SELECT count(*) FROM annotations"
                    " WHERE image_id = %s AND source = 'humain'",
                    (ctx["i_prop"],)) == 0
    assert exec_sql(engine, "SELECT count(*) FROM annotations"
                    " WHERE image_id = %s", (ctx["i_prop"],)) == 2


def test_voisines(make_client, ctx, engine):
    """Ordre par id dans le lot, images archivées sautées, bornes à null."""
    ids = {"s1": ctx["s1"], "b1": ctx["b1"]}
    i_archivee = poser_image(engine, ids, "arch.jpg",
                             superseded_at=datetime.now(timezone.utc))
    i_fin = poser_image(engine, ids, "fin.jpg")

    alice = login(make_client(), "alice")
    r = alice.get(f"/api/images/{ctx['i_file']}/voisines")
    assert r.status_code == 200
    corps = r.json()
    assert corps["precedente"]["id"] == ctx["i_vierge"]
    assert corps["precedente"]["statut"] == "a_annoter"
    # l'archivée entre i_file et i_fin est sautée
    assert i_archivee > ctx["i_file"] and i_archivee < i_fin
    assert corps["suivante"]["id"] == i_fin
    assert corps["suivante"]["nom"] == "fin.jpg"
    # bornes du lot
    assert alice.get(f"/api/images/{ctx['i_prop']}/voisines"
                     ).json()["precedente"] is None
    assert alice.get(f"/api/images/{i_fin}/voisines"
                     ).json()["suivante"] is None
    assert make_client().get(
        f"/api/images/{i_fin}/voisines").status_code == 401


def test_avancement(make_client, ctx, engine):
    alice = login(make_client(), "alice")
    alice.post(f"/api/images/{ctx['i_prop']}/ouvrir")  # -> en_cours

    r = alice.get(f"/api/batches/{ctx['b1']}/avancement")
    assert r.status_code == 200
    corps = r.json()
    assert corps["batch_id"] == ctx["b1"] and corps["total"] == 3
    assert corps["par_statut"] == {
        "en_attente_preannotation": 1, "a_annoter": 1, "en_cours": 1,
        "annotee": 0, "relue": 0}

    # bob ne détient pas le lot ; l'admin voit tout ; lot inconnu
    assert login(make_client(), "bob").get(
        f"/api/batches/{ctx['b1']}/avancement").status_code == 403
    assert login(make_client(), "root").get(
        f"/api/batches/{ctx['b1']}/avancement").status_code == 200
    assert alice.get("/api/batches/99999/avancement").status_code == 404
