"""Assignation, verrou de lot (y compris concurrence réelle), découpage."""
import threading
import time
from datetime import date

import psycopg
import pytest

from conftest import exec_sql, exec_sql_all, insert_image


@pytest.fixture
def ctx(db, make_user):
    from app.models import Batch, CaptureSession

    admin = make_user("root", role="administrateur")
    alice = make_user("alice")
    bob = make_user("bob", role="annotateur_confirme")
    session = CaptureSession(name="s1", captured_on=date(2026, 7, 1),
                             created_by=admin.id)
    db.add(session)
    db.flush()
    batch = Batch(session_id=session.id, name="import", created_by=admin.id)
    db.add(batch)
    db.commit()
    return {"admin": admin, "alice": alice, "bob": bob,
            "s1": session.id, "b1": batch.id}


def login(client, username, password="motdepasse123"):
    r = client.post("/api/auth/login",
                    json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return client


def test_assignation_et_verrou_api(make_client, ctx):
    admin = login(make_client(), "root")
    r = admin.post(f"/api/batches/{ctx['b1']}/assign",
                   json={"user_id": ctx["alice"].id})
    assert r.status_code == 201
    # verrou : second annotateur refusé, détenteur nommé
    r = admin.post(f"/api/batches/{ctx['b1']}/assign",
                   json={"user_id": ctx["bob"].id})
    assert r.status_code == 409
    assert "alice" in r.json()["detail"]
    # visible dans la liste, avec la date de prise du verrou
    lots = login(make_client(), "alice").get("/api/batches").json()
    assert lots[0]["holder"] == "alice" and lots[0]["session_name"] == "s1"
    assert lots[0]["holder_since"] is not None
    # libération admin (force) puis réassignation possible
    r = admin.post(f"/api/batches/{ctx['b1']}/release", json={})
    assert r.status_code == 200 and r.json()["reason"] == "force_admin"
    assert admin.post(f"/api/batches/{ctx['b1']}/assign",
                      json={"user_id": ctx["bob"].id}).status_code == 201


def test_gardes_assignation(make_client, ctx, engine):
    exec_sql(engine, "UPDATE users SET is_active = false WHERE id = %s",
             (ctx["bob"].id,))
    admin = login(make_client(), "root")
    assert admin.post("/api/batches/9999/assign",
                      json={"user_id": ctx["alice"].id}).status_code == 404
    assert admin.post(f"/api/batches/{ctx['b1']}/assign",
                      json={"user_id": ctx["bob"].id}).status_code == 400
    annotateur = login(make_client(), "alice")
    assert annotateur.post(f"/api/batches/{ctx['b1']}/assign",
                           json={"user_id": ctx["alice"].id}).status_code == 403


def test_liberation_par_detenteur_et_gardes(make_client, ctx, engine):
    admin = login(make_client(), "root")
    admin.post(f"/api/batches/{ctx['b1']}/assign",
               json={"user_id": ctx["alice"].id})
    # un autre annotateur ne libère pas
    bob = login(make_client(), "bob")
    assert bob.post(f"/api/batches/{ctx['b1']}/release", json={}).status_code == 403
    # le détenteur libère (motif par défaut : rendu)
    alice = login(make_client(), "alice")
    r = alice.post(f"/api/batches/{ctx['b1']}/release", json={})
    assert r.status_code == 200 and r.json()["reason"] == "rendu"
    assert exec_sql(engine,
                    "SELECT release_reason FROM batch_assignments"
                    " WHERE batch_id = %s", (ctx["b1"],)) == "rendu"
    # plus d'assignation active
    assert alice.post(f"/api/batches/{ctx['b1']}/release",
                      json={}).status_code == 404
    # motif « terminé » explicite
    admin.post(f"/api/batches/{ctx['b1']}/assign",
               json={"user_id": ctx["alice"].id})
    r = alice.post(f"/api/batches/{ctx['b1']}/release",
                   json={"reason": "termine"})
    assert r.json()["reason"] == "termine"


def test_liberation_remet_les_en_cours_a_annoter(make_client, ctx, engine):
    ids = {"s1": ctx["s1"], "b1": ctx["b1"]}
    en_cours, autre, annotee = (
        insert_image(engine, ids, f"{i}" * 64, f"p/{i}.jpg") for i in "abc"
    )
    for img in (en_cours, autre, annotee):
        exec_sql(engine, "UPDATE images SET status = 'a_annoter' WHERE id = %s", (img,))
        exec_sql(engine, "UPDATE images SET status = 'en_cours' WHERE id = %s", (img,))
    exec_sql(engine, "UPDATE images SET status = 'annotee' WHERE id = %s", (annotee,))

    admin = login(make_client(), "root")
    admin.post(f"/api/batches/{ctx['b1']}/assign",
               json={"user_id": ctx["alice"].id})
    alice = login(make_client(), "alice")
    r = alice.post(f"/api/batches/{ctx['b1']}/release", json={})
    assert r.json()["reverted_images"] == 2
    assert exec_sql(engine,
                    "SELECT count(*) FROM images WHERE status = 'a_annoter'") == 2
    assert exec_sql(engine, "SELECT status FROM images WHERE id = %s",
                    (annotee,)) == "annotee"
    # transitions tracées, avec leur auteur
    assert exec_sql(engine,
                    "SELECT count(*) FROM image_status_events"
                    " WHERE from_status = 'en_cours' AND to_status = 'a_annoter'"
                    " AND changed_by = %s", (ctx["alice"].id,)) == 2


def test_liberation_rend_chaque_image_a_son_origine(make_client, ctx, engine):
    """La libération rend chaque image « en cours » à son statut d'ORIGINE,
    lu dans le journal (même règle que /images/{id}/fermer) : une image
    annotée rouverte reste annotée — la rétrograder la sortirait de l'export,
    donc de l'entraînement, sans aucun signal — et une image jamais annotée
    revient à annoter."""
    ids = {"s1": ctx["s1"], "b1": ctx["b1"]}
    rouverte = insert_image(engine, ids, "d" * 64, "p/d.jpg")
    vierge = insert_image(engine, ids, "f" * 64, "p/f.jpg")
    admin = login(make_client(), "root")
    admin.post("/api/images/passer-a-annoter",
               json={"image_ids": [rouverte, vierge]})
    admin.post(f"/api/batches/{ctx['b1']}/assign",
               json={"user_id": ctx["alice"].id})

    # alice annote la première (négatif) puis la ROUVRE ; elle ouvre la
    # seconde sans jamais l'enregistrer
    alice = login(make_client(), "alice")
    alice.post(f"/api/images/{rouverte}/ouvrir")
    assert alice.put(f"/api/images/{rouverte}/annotations",
                     json={"boites": []}).status_code == 200
    alice.post(f"/api/images/{rouverte}/ouvrir")
    alice.post(f"/api/images/{vierge}/ouvrir")

    r = alice.post(f"/api/batches/{ctx['b1']}/release", json={})
    assert r.status_code == 200 and r.json()["reverted_images"] == 2
    assert exec_sql(engine, "SELECT status FROM images WHERE id = %s",
                    (rouverte,)) == "annotee"
    assert exec_sql(engine, "SELECT status FROM images WHERE id = %s",
                    (vierge,)) == "a_annoter"
    # restitutions tracées avec alice comme auteur (2 en_cours → annotee :
    # l'enregistrement, puis la restitution à la libération)
    assert exec_sql(engine, "SELECT count(*) FROM image_status_events"
                    " WHERE image_id = %s AND from_status = 'en_cours'"
                    " AND to_status = 'annotee' AND changed_by = %s",
                    (rouverte, ctx["alice"].id)) == 2
    assert exec_sql(engine, "SELECT count(*) FROM image_status_events"
                    " WHERE image_id = %s AND from_status = 'en_cours'"
                    " AND to_status = 'a_annoter' AND changed_by = %s",
                    (vierge, ctx["alice"].id)) == 1


def test_verrou_sous_assignations_concurrentes(engine, ctx):
    """Deux transactions SIMULTANÉES tentent d'assigner le même lot : la
    seconde se bloque sur l'index unique partiel jusqu'au commit de la
    première, puis est refusée. C'est PostgreSQL qui arbitre, pas l'API."""
    url = engine.url
    dsn = f"host={url.host} port={url.port} user={url.username} dbname={url.database}"
    b1, admin = ctx["b1"], ctx["admin"].id
    resultats = []

    premiere = psycopg.connect(dsn)
    premiere.execute(
        "INSERT INTO batch_assignments (batch_id, user_id, assigned_by)"
        " VALUES (%s, %s, %s)", (b1, ctx["alice"].id, admin))  # non commitée

    def rivale():
        with psycopg.connect(dsn) as conn:
            try:
                conn.execute(
                    "INSERT INTO batch_assignments (batch_id, user_id, assigned_by)"
                    " VALUES (%s, %s, %s)", (b1, ctx["bob"].id, admin))
                conn.commit()
                resultats.append("inseree")
            except psycopg.errors.UniqueViolation:
                resultats.append("refusee")

    t = threading.Thread(target=rivale)
    t.start()
    time.sleep(0.5)  # laisser la rivale se bloquer sur l'index
    premiere.commit()
    premiere.close()
    t.join(timeout=10)

    assert resultats == ["refusee"]
    assert exec_sql(engine,
                    "SELECT count(*) FROM batch_assignments"
                    " WHERE released_at IS NULL") == 1
    assert exec_sql(engine,
                    "SELECT user_id FROM batch_assignments"
                    " WHERE released_at IS NULL") == ctx["alice"].id


def test_decoupage_en_lots_de_n(make_client, ctx, engine):
    ids = {"s1": ctx["s1"], "b1": ctx["b1"]}
    for i in range(5):
        insert_image(engine, ids, f"{i}" * 64, f"p/{i}.jpg")
    entamee = insert_image(engine, ids, "e" * 64, "p/e.jpg")
    exec_sql(engine, "UPDATE images SET status = 'a_annoter' WHERE id = %s", (entamee,))
    exec_sql(engine, "UPDATE images SET status = 'en_cours' WHERE id = %s", (entamee,))
    exec_sql(engine, "UPDATE images SET status = 'annotee' WHERE id = %s", (entamee,))

    admin = login(make_client(), "root")
    r = admin.post(f"/api/batches/{ctx['b1']}/split", json={"size": 2})
    assert r.status_code == 200
    corps = r.json()
    assert [b["count"] for b in corps["created"]] == [2, 2, 1]
    assert [b["name"] for b in corps["created"]] == ["lot 1", "lot 2", "lot 3"]
    assert corps["moved"] == 5
    # l'image entamée reste dans le lot d'origine
    assert exec_sql(engine, "SELECT batch_id FROM images WHERE id = %s",
                    (entamee,)) == ctx["b1"]
    # plus rien à découper dans le backlog
    assert admin.post(f"/api/batches/{ctx['b1']}/split",
                      json={"size": 2}).status_code == 400


def test_avancement_compte_les_images_garees(make_client, ctx, engine):
    """Une image garée (échec de pré-annotation au plafond) a le statut
    en_attente_preannotation mais ne progressera plus sans geste admin : la
    compter à part évite d'attendre indéfiniment une file morte."""
    ids = {"s1": ctx["s1"], "b1": ctx["b1"]}
    insert_image(engine, ids, "a" * 64, "p/a.jpg")
    insert_image(engine, ids, "b" * 64, "p/b.jpg",
                 preannotation_attempts=3,
                 preannotation_error="fichier tronqué",
                 preannotation_error_kind="fichier_illisible")

    lots = login(make_client(), "root").get("/api/batches").json()
    assert lots[0]["total_images"] == 2
    assert lots[0]["parked_images"] == 1
    assert lots[0]["done_images"] == 0


def test_decoupage_round_robin_entre_postes(make_client, ctx, engine):
    """Le découpage alterne les dossiers sources : un lot homogène par poste
    de capture corrélerait le style d'un annotateur à un appareil. Ici les
    images sont insérées poste par poste (l'ordre par id donnerait des lots
    homogènes) — chaque lot doit contenir un posteA ET un posteB."""
    ids = {"s1": ctx["s1"], "b1": ctx["b1"]}
    for i in range(3):
        insert_image(engine, ids, f"{i}" * 64, f"posteA/a{i}.jpg")
    for i in range(3, 6):
        insert_image(engine, ids, f"{i}" * 64, f"posteB/b{i}.jpg")

    admin = login(make_client(), "root")
    r = admin.post(f"/api/batches/{ctx['b1']}/split", json={"size": 2})
    assert r.status_code == 200
    assert [b["count"] for b in r.json()["created"]] == [2, 2, 2]
    for lot in r.json()["created"]:
        labels = exec_sql_all(engine,
                              "SELECT source_label FROM images"
                              " WHERE batch_id = %s", (lot["id"],))
        assert sorted(row[0] for row in labels) == ["posteA", "posteB"]


def test_decoupage_refuse_si_verrouille_et_numerote_sans_collision(
        make_client, ctx, engine, db):
    from app.models import Batch

    ids = {"s1": ctx["s1"], "b1": ctx["b1"]}
    insert_image(engine, ids, "f" * 64, "p/f.jpg")
    db.add(Batch(session_id=ctx["s1"], name="lot 1", created_by=ctx["admin"].id))
    db.commit()

    admin = login(make_client(), "root")
    admin.post(f"/api/batches/{ctx['b1']}/assign",
               json={"user_id": ctx["alice"].id})
    assert admin.post(f"/api/batches/{ctx['b1']}/split",
                      json={"size": 10}).status_code == 409
    admin.post(f"/api/batches/{ctx['b1']}/release", json={})
    r = admin.post(f"/api/batches/{ctx['b1']}/split", json={"size": 10})
    # « lot 1 » existait déjà dans la session : la numérotation saute
    assert r.json()["created"][0]["name"] == "lot 2"
