"""API du canvas (segment 1) : droits par verrou de lot, transitions de
statut tracées, enregistrement à état complet (propositions toutes tranchées,
rejetées conservées, négatif valide), navigation, avancement."""
import itertools
from datetime import date, datetime, timezone

import pytest

from conftest import STORAGE_TEST_ROOT, exec_sql, exec_sql_all, insert_image
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
    assert corps["deja_annotee"] is False  # jamais annotée à ce stade
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
    # déjà annotée : le signal SURVIT à la réouverture (le statut, non) —
    # c'est lui qui distingue un négatif validé d'une image jamais regardée
    assert r.json()["deja_annotee"] is True
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


def test_fermer_image(make_client, ctx, engine):
    """Fermeture d'une image quittée sans enregistrement : retour à son
    statut d'origine lu dans le journal (annotee si elle l'était, a_annoter
    sinon), événement porté par l'utilisateur, annotations intactes."""
    # droits : non authentifié, puis compte sans verrou sur le lot
    assert make_client().post(
        f"/api/images/{ctx['i_vierge']}/fermer").status_code == 401
    assert login(make_client(), "bob").post(
        f"/api/images/{ctx['i_vierge']}/fermer").status_code == 403
    alice = login(make_client(), "alice")
    # image inconnue
    assert alice.post("/api/images/999999/fermer").status_code == 404
    # pas en_cours : rien à refermer
    assert alice.post(
        f"/api/images/{ctx['i_vierge']}/fermer").status_code == 409

    # jamais annotée : ouvrir puis fermer → retour a_annoter, événement
    # porté par alice (jamais NULL — NULL est la signature du worker)
    alice.post(f"/api/images/{ctx['i_vierge']}/ouvrir")
    r = alice.post(f"/api/images/{ctx['i_vierge']}/fermer")
    assert r.status_code == 200 and r.json() == {"statut": "a_annoter"}
    assert exec_sql(engine, "SELECT status FROM images WHERE id = %s",
                    (ctx["i_vierge"],)) == "a_annoter"
    assert exec_sql(engine, "SELECT changed_by FROM image_status_events"
                    " WHERE image_id = %s AND from_status = 'en_cours'"
                    " AND to_status = 'a_annoter'",
                    (ctx["i_vierge"],)) == ctx["alice"].id
    # refermée : plus comptée « en cours » dans l'écran des lots
    lots = alice.get("/api/batches").json()
    assert [l["in_progress_images"] for l in lots if l["id"] == ctx["b1"]] == [0]

    # déjà annotée : enregistrer, rouvrir, fermer → retour annotee ;
    # annotated_at intact (une fermeture n'est pas un enregistrement)
    alice.post(f"/api/images/{ctx['i_vierge']}/ouvrir")
    assert alice.put(f"/api/images/{ctx['i_vierge']}/annotations",
                     json={"boites": []}).status_code == 200
    annote_a = exec_sql(engine, "SELECT annotated_at FROM images"
                        " WHERE id = %s", (ctx["i_vierge"],))
    alice.post(f"/api/images/{ctx['i_vierge']}/ouvrir")
    r = alice.post(f"/api/images/{ctx['i_vierge']}/fermer")
    assert r.status_code == 200 and r.json() == {"statut": "annotee"}
    assert exec_sql(engine, "SELECT status FROM images WHERE id = %s",
                    (ctx["i_vierge"],)) == "annotee"
    assert exec_sql(engine, "SELECT annotated_at FROM images WHERE id = %s",
                    (ctx["i_vierge"],)) == annote_a

    # les propositions non tranchées restent intactes (fermer ≠ trancher)
    alice.post(f"/api/images/{ctx['i_prop']}/ouvrir")
    assert alice.post(f"/api/images/{ctx['i_prop']}/fermer").status_code == 200
    assert exec_sql(engine, "SELECT count(*) FROM annotations"
                    " WHERE image_id = %s AND state = 'proposee'",
                    (ctx["i_prop"],)) == 2

    # origine « relue » : restituée telle quelle, relecteur intact —
    # en_cours → relue est entrée dans la liste blanche (migration 0004)
    exec_sql(engine, "UPDATE images SET status = 'relue', reviewed_by = %s,"
             " reviewed_at = now() WHERE id = %s RETURNING id",
             (ctx["admin"].id, ctx["i_vierge"]))
    alice.post(f"/api/images/{ctx['i_vierge']}/ouvrir")
    r = alice.post(f"/api/images/{ctx['i_vierge']}/fermer")
    assert r.status_code == 200 and r.json() == {"statut": "relue"}
    assert exec_sql(engine, "SELECT status || '/' || reviewed_by::text"
                    " FROM images WHERE id = %s",
                    (ctx["i_vierge"],)) == f"relue/{ctx['admin'].id}"


def test_relire(make_client, ctx, engine, make_user):
    """Relecture : un annotateur confirmé (ou un administrateur) corrige ou
    valide en l'état une image annotée par QUELQU'UN D'AUTRE — elle passe
    `relue` avec relecteur et horodatage, l'auteur de l'annotation ne bouge
    pas. en_cours → relue est entrée dans la liste blanche (migration 0004).
    Une réannotation ultérieure PÉRIME la relecture."""
    carla = make_user("carla", role="annotateur_confirme")

    # alice annote (négatif) puis rouvre : l'image reste en_cours
    alice = login(make_client(), "alice")
    alice.post(f"/api/images/{ctx['i_vierge']}/ouvrir")
    assert alice.put(f"/api/images/{ctx['i_vierge']}/annotations",
                     json={"boites": []}).status_code == 200
    alice.post(f"/api/images/{ctx['i_vierge']}/ouvrir")
    # rôle insuffisant : un annotateur simple ne relit pas
    assert alice.post(f"/api/images/{ctx['i_vierge']}/relire",
                      json={"boites": []}).status_code == 403
    assert make_client().post(f"/api/images/{ctx['i_vierge']}/relire",
                              json={"boites": []}).status_code == 401

    # le lot passe à carla (relire exige les mêmes droits d'accès qu'ouvrir)
    exec_sql(engine, "UPDATE batch_assignments SET released_at = now(),"
             " released_by = %s, release_reason = 'force_admin'"
             " WHERE batch_id = %s AND released_at IS NULL RETURNING id",
             (ctx["admin"].id, ctx["b1"]))
    exec_sql(engine,
             "INSERT INTO batch_assignments (batch_id, user_id, assigned_by)"
             " VALUES (%s, %s, %s) RETURNING id",
             (ctx["b1"], carla.id, ctx["admin"].id))
    relectrice = login(make_client(), "carla")
    assert relectrice.post("/api/images/999999/relire",
                           json={"boites": []}).status_code == 404

    # l'ouverture annonce qui a annoté et quand — la relecture sait qui elle
    # relit, et le front en déduit si le bouton s'affiche
    r = relectrice.post(f"/api/images/{ctx['i_vierge']}/ouvrir")
    assert r.status_code == 200
    corps = r.json()
    assert corps["annotee_par_id"] == ctx["alice"].id
    assert corps["annotee_par"] == "alice" and corps["annotee_le"] is not None
    assert corps["relue_par"] is None and corps["relue_le"] is None

    # correction en relecture : une boîte ajoutée, l'image passe relue
    r = relectrice.post(f"/api/images/{ctx['i_vierge']}/relire", json={
        "boites": [{"class_id": 6, "x_center": 0.5, "y_center": 0.5,
                    "box_width": 0.2, "box_height": 0.2, "etat": "validee"}]})
    assert r.status_code == 200, r.text
    assert r.json() == {"statut": "relue", "validees": 1, "rejetees": 0,
                        "supprimees": 0}
    assert exec_sql(engine, "SELECT status || '/' || reviewed_by::text"
                    " FROM images WHERE id = %s",
                    (ctx["i_vierge"],)) == f"relue/{carla.id}"
    # l'auteur de l'annotation n'a pas bougé ; la retouche est au relecteur
    assert exec_sql(engine, "SELECT annotated_by FROM images WHERE id = %s",
                    (ctx["i_vierge"],)) == ctx["alice"].id
    assert exec_sql(engine, "SELECT created_by FROM annotations"
                    " WHERE image_id = %s AND class_id = 6",
                    (ctx["i_vierge"],)) == carla.id
    assert exec_sql(engine, "SELECT changed_by FROM image_status_events"
                    " WHERE image_id = %s AND from_status = 'en_cours'"
                    " AND to_status = 'relue'", (ctx["i_vierge"],)) == carla.id
    # pas en_cours : l'ouvrir d'abord
    assert relectrice.post(f"/api/images/{ctx['i_vierge']}/relire",
                           json={"boites": []}).status_code == 409
    # et l'ouverture annonce maintenant la relecture en vigueur
    r = relectrice.post(f"/api/images/{ctx['i_vierge']}/ouvrir")
    assert r.json()["relue_par"] == "carla" and r.json()["relue_le"] is not None

    # une réannotation PÉRIME la relecture : carla réenregistre — l'image
    # redevient annotee (par carla) et le relecteur est effacé, sinon la
    # contrainte relecteur ≠ annotateur exploserait
    bid = r.json()["boites"][0]["id"]
    r = relectrice.put(f"/api/images/{ctx['i_vierge']}/annotations", json={
        "boites": [{"id": bid, "class_id": 6, "x_center": 0.5,
                    "y_center": 0.5, "box_width": 0.2, "box_height": 0.2,
                    "etat": "validee"}]})
    assert r.status_code == 200
    assert exec_sql(engine, "SELECT status || '/' || annotated_by::text"
                    " FROM images WHERE id = %s",
                    (ctx["i_vierge"],)) == f"annotee/{carla.id}"
    assert exec_sql(engine, "SELECT reviewed_by FROM images WHERE id = %s",
                    (ctx["i_vierge"],)) is None

    # on ne relit pas sa propre annotation
    relectrice.post(f"/api/images/{ctx['i_vierge']}/ouvrir")
    r = relectrice.post(f"/api/images/{ctx['i_vierge']}/relire",
                        json={"boites": []})
    assert r.status_code == 409 and "propre annotation" in r.json()["detail"]

    # une image jamais annotée n'a rien à relire
    relectrice.post(f"/api/images/{ctx['i_prop']}/ouvrir")
    r = relectrice.post(f"/api/images/{ctx['i_prop']}/relire",
                        json={"boites": []})
    assert r.status_code == 409 and "jamais annotée" in r.json()["detail"]


def test_passer_a_annoter(make_client, ctx, engine):
    """Court-circuit admin de la file : tout-ou-rien, événement porté par
    l'admin (jamais NULL — NULL est la signature du worker), motifs d'échec
    laissés en place."""
    # droits : non authentifié, puis annotateur (même détenteur du lot)
    assert make_client().post("/api/images/passer-a-annoter", json={
        "image_ids": [ctx["i_file"]]}).status_code == 401
    assert login(make_client(), "alice").post(
        "/api/images/passer-a-annoter",
        json={"image_ids": [ctx["i_file"]]}).status_code == 403

    admin = login(make_client(), "root")
    # tout-ou-rien : une image hors file fait échouer le lot entier
    r = admin.post("/api/images/passer-a-annoter", json={
        "image_ids": [ctx["i_file"], ctx["i_vierge"]]})
    assert r.status_code == 409 and str(ctx["i_vierge"]) in r.json()["detail"]
    assert exec_sql(engine, "SELECT status FROM images WHERE id = %s",
                    (ctx["i_file"],)) == "en_attente_preannotation"
    # id inconnu
    assert admin.post("/api/images/passer-a-annoter", json={
        "image_ids": [ctx["i_file"], 999999]}).status_code == 404
    # liste vide refusée par le schéma
    assert admin.post("/api/images/passer-a-annoter",
                      json={"image_ids": []}).status_code == 422

    # nominal, sur une image « garée » : les motifs d'échec restent en place
    exec_sql(engine, "UPDATE images SET preannotation_attempts = 3,"
             " preannotation_error = 'poids illisibles',"
             " preannotation_error_kind = 'moteur_indisponible'"
             " WHERE id = %s RETURNING id", (ctx["i_file"],))
    r = admin.post("/api/images/passer-a-annoter",
                   json={"image_ids": [ctx["i_file"]]})
    assert r.status_code == 200 and r.json() == {"passees": 1}
    assert exec_sql(engine, "SELECT status FROM images WHERE id = %s",
                    (ctx["i_file"],)) == "a_annoter"
    assert exec_sql(engine, "SELECT preannotation_attempts FROM images"
                    " WHERE id = %s", (ctx["i_file"],)) == 3
    # événement tracé avec l'admin comme auteur, pas NULL
    assert exec_sql(engine, "SELECT changed_by FROM image_status_events"
                    " WHERE image_id = %s"
                    " AND from_status = 'en_attente_preannotation'"
                    " AND to_status = 'a_annoter'",
                    (ctx["i_file"],)) == ctx["admin"].id
    # l'image sortie de la file n'est plus comptée « garée »
    lots = admin.get("/api/batches").json()
    assert [l["parked_images"] for l in lots if l["id"] == ctx["b1"]] == [0]

    # rejouer le même geste : l'image n'est plus en file
    assert admin.post("/api/images/passer-a-annoter", json={
        "image_ids": [ctx["i_file"]]}).status_code == 409
    # image archivée : introuvable
    exec_sql(engine, "UPDATE images SET superseded_at = %s WHERE id = %s"
             " RETURNING id", (datetime.now(timezone.utc), ctx["i_vierge"]))
    assert admin.post("/api/images/passer-a-annoter", json={
        "image_ids": [ctx["i_vierge"]]}).status_code == 404


def test_images_du_lot(make_client, ctx, engine):
    """Liste ordonnée par id, archivées exclues, propositions comptées —
    mêmes droits que l'avancement."""
    ids = {"s1": ctx["s1"], "b1": ctx["b1"]}
    poser_image(engine, ids, "arch2.jpg",
                superseded_at=datetime.now(timezone.utc))

    assert make_client().get(
        f"/api/batches/{ctx['b1']}/images").status_code == 401
    assert login(make_client(), "bob").get(
        f"/api/batches/{ctx['b1']}/images").status_code == 403
    assert login(make_client(), "root").get(
        f"/api/batches/{ctx['b1']}/images").status_code == 200
    assert login(make_client(), "root").get(
        "/api/batches/99999/images").status_code == 404

    alice = login(make_client(), "alice")
    corps = alice.get(f"/api/batches/{ctx['b1']}/images").json()
    assert [(i["nom"], i["statut"], i["propositions"]) for i in corps] == [
        ("prop.jpg", "a_annoter", 2),
        ("vierge.jpg", "a_annoter", 0),
        ("file.jpg", "en_attente_preannotation", 0),
    ]
    assert [i["id"] for i in corps] == sorted(i["id"] for i in corps)
    # classes présentes (proposées ou validées) : la trame du filtre
    assert [i["classes"] for i in corps] == [[0, 1], [], []]

    # une proposition tranchée ne compte plus
    alice.post(f"/api/images/{ctx['i_prop']}/ouvrir")
    alice.put(f"/api/images/{ctx['i_prop']}/annotations", json={"boites": [
        {"id": ctx["p1"], "class_id": 0, "x_center": 0.3, "y_center": 0.3,
         "box_width": 0.1, "box_height": 0.1, "etat": "validee"},
        {"id": ctx["p2"], "class_id": 1, "x_center": 0.6, "y_center": 0.6,
         "box_width": 0.2, "box_height": 0.2, "etat": "rejetee"},
    ]})
    corps = alice.get(f"/api/batches/{ctx['b1']}/images").json()
    assert (corps[0]["statut"], corps[0]["propositions"]) == ("annotee", 0)
    # la rejetée (classe 1) ne « porte » plus sa classe — faux positif
    assert corps[0]["classes"] == [0]


def test_referentiel_classes(make_client, ctx):
    """GET /api/images/classes : le référentiel data.yaml pour les écrans
    sans image ouverte (filtre de la liste). Authentifié, sans rôle requis."""
    assert make_client().get("/api/images/classes").status_code == 401
    r = login(make_client(), "alice").get("/api/images/classes")
    assert r.status_code == 200
    classes = r.json()["classes"]
    assert classes[0] == "Plastique" and len(classes) == 8


@pytest.fixture
def reduction_128():
    """Force un côté max de 128 px pour tester la réduction sur de petits
    fichiers ; restaure la configuration en sortie."""
    import os

    from app.config import get_settings

    os.environ["REDUCTION_MAX_COTE"] = "128"
    get_settings.cache_clear()
    yield
    os.environ.pop("REDUCTION_MAX_COTE", None)
    get_settings.cache_clear()


def test_fichier_reduit_et_coordonnees_stables(make_client, ctx, engine,
                                               reduction_128):
    """Au-delà du côté max, /fichier sert une réduction JPEG mise en cache
    (répertoire dédié) ; l'original reste intact et accessible ; et surtout
    LES COORDONNÉES NE BOUGENT PAS : normalisées, elles se rapportent aux
    dimensions du fichier annoté, jamais à celles de la réduction."""
    import io as io_

    from PIL import Image as PILImage

    from app.storage import get_storage

    ids = {"s1": ctx["s1"], "b1": ctx["b1"]}
    dossier = STORAGE_TEST_ROOT / "canvas"
    dossier.mkdir(parents=True, exist_ok=True)
    PILImage.new("RGB", (200, 150), "green").save(dossier / "grande.jpg")
    sha = f"{next(_sha):02x}" * 32
    i_grande = insert_image(engine, ids, sha, "canvas/grande.jpg",
                            status="a_annoter", width=200, height=150)

    alice = login(make_client(), "alice")
    r = alice.get(f"/api/images/{i_grande}/fichier")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert PILImage.open(io_.BytesIO(r.content)).size == (128, 96)
    # cache créé dans son répertoire dédié, resservi tel quel
    cache_dir = STORAGE_TEST_ROOT / ".cache_reduites"
    fichiers = sorted(cache_dir.iterdir())
    assert len(fichiers) == 1
    assert alice.get(f"/api/images/{i_grande}/fichier").content == r.content
    assert sorted(cache_dir.iterdir()) == fichiers
    # l'original n'est jamais modifié et reste accessible
    ro = alice.get(f"/api/images/{i_grande}/fichier?original=true")
    assert ro.content == get_storage().read("canvas/grande.jpg")
    # sous le seuil : servie telle quelle, aucune entrée de cache
    rp = alice.get(f"/api/images/{ctx['i_vierge']}/fichier")
    assert rp.content == get_storage().read("canvas/vierge.jpg")
    assert sorted(cache_dir.iterdir()) == fichiers

    # invariance des coordonnées : dimensions annoncées = fichier annoté,
    # aller-retour exact des valeurs normalisées, réduction ou pas
    corps = alice.post(f"/api/images/{i_grande}/ouvrir").json()
    assert (corps["largeur"], corps["hauteur"]) == (200, 150)
    r = alice.put(f"/api/images/{i_grande}/annotations", json={"boites": [
        {"class_id": 5, "x_center": 0.25, "y_center": 0.4,
         "box_width": 0.5, "box_height": 0.25, "etat": "validee"}]})
    assert r.status_code == 200
    ligne = exec_sql_all(engine, "SELECT x_center, y_center, box_width,"
                         " box_height FROM annotations WHERE image_id = %s",
                         (i_grande,))
    assert list(ligne[0]) == [0.25, 0.4, 0.5, 0.25]
    corps = alice.post(f"/api/images/{i_grande}/ouvrir").json()
    boite = corps["boites"][0]
    assert (boite["x_center"], boite["y_center"], boite["box_width"],
            boite["box_height"]) == (0.25, 0.4, 0.5, 0.25)


def test_avancement(make_client, ctx, engine):
    alice = login(make_client(), "alice")
    alice.post(f"/api/images/{ctx['i_prop']}/ouvrir")  # -> en_cours

    # les lots comptent l'en_cours À PART : l'avancement ne recule pas
    # parce qu'on a ouvert une image
    lots = alice.get("/api/batches").json()
    lot = next(l for l in lots if l["id"] == ctx["b1"])
    assert (lot["done_images"], lot["in_progress_images"]) == (0, 1)

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
