"""Écran d'administration technique : liste des sessions, état de la file de
pré-annotation, relance des images garées. Administrateur uniquement — le
worker n'est pas piloté d'ici, l'écran observe sa file."""
from datetime import date

import pytest

from conftest import exec_sql, insert_image


@pytest.fixture
def ctx(db, engine, make_user):
    from app.models import Batch, CaptureSession

    admin = make_user("root", role="administrateur")
    alice = make_user("alice")
    session = CaptureSession(name="s1", captured_on=date(2026, 7, 1),
                             created_by=admin.id)
    db.add(session)
    db.flush()
    batch = Batch(session_id=session.id, name="import", created_by=admin.id)
    db.add(batch)
    db.commit()
    return {"admin": admin, "alice": alice,
            "s1": session.id, "b1": batch.id}


def login(client, username, password="motdepasse123"):
    r = client.post("/api/auth/login",
                    json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return client


def test_sessions_admin_seulement(make_client, ctx, engine):
    ids = {"s1": ctx["s1"], "b1": ctx["b1"]}
    insert_image(engine, ids, "a" * 64, "p/a.jpg")

    assert make_client().get("/api/sessions").status_code == 401
    assert login(make_client(), "alice").get("/api/sessions").status_code == 403
    r = login(make_client(), "root").get("/api/sessions")
    assert r.status_code == 200
    assert [(s["name"], s["captured_on"], s["images"]) for s in r.json()] == [
        ("s1", "2026-07-01", 1)]


def test_etat_file_et_relance(make_client, ctx, engine):
    """La file distingue « en attente » (le worker les prendra) des
    « garées » (au plafond de tentatives, motif affiché) ; la relance remet
    le compteur à zéro SANS toucher au statut (aucune transition) ni aux
    motifs (ils documentent le dernier échec jusqu'au prochain passage)."""
    ids = {"s1": ctx["s1"], "b1": ctx["b1"]}
    en_file = insert_image(engine, ids, "b" * 64, "p/b.jpg")
    garee = insert_image(engine, ids, "c" * 64, "p/c.jpg",
                         preannotation_attempts=3,
                         preannotation_error="fichier tronqué",
                         preannotation_error_kind="fichier_illisible")
    hors = insert_image(engine, ids, "d" * 64, "p/d.jpg", status="a_annoter")

    assert make_client().get("/api/preannotation/etat").status_code == 401
    assert login(make_client(), "alice").get(
        "/api/preannotation/etat").status_code == 403
    admin = login(make_client(), "root")
    corps = admin.get("/api/preannotation/etat").json()
    assert corps["en_attente"] == 1 and corps["plafond_tentatives"] == 3
    assert [(g["id"], g["tentatives"], g["motif_type"], g["motif"])
            for g in corps["garees"]] == [
        (garee, 3, "fichier_illisible", "fichier tronqué")]
    assert corps["garees"][0]["lot"] == "import"
    assert corps["garees"][0]["session"] == "s1"

    # relance : tout-ou-rien, comme le court-circuit
    assert login(make_client(), "alice").post(
        "/api/preannotation/relancer",
        json={"image_ids": [garee]}).status_code == 403
    assert admin.post("/api/preannotation/relancer", json={
        "image_ids": [garee, 999999]}).status_code == 404
    assert admin.post("/api/preannotation/relancer", json={
        "image_ids": [garee, hors]}).status_code == 409
    assert exec_sql(engine, "SELECT preannotation_attempts FROM images"
                    " WHERE id = %s", (garee,)) == 3
    assert admin.post("/api/preannotation/relancer",
                      json={"image_ids": []}).status_code == 422

    r = admin.post("/api/preannotation/relancer", json={"image_ids": [garee]})
    assert r.status_code == 200 and r.json() == {"relancees": 1}
    assert exec_sql(engine, "SELECT preannotation_attempts FROM images"
                    " WHERE id = %s", (garee,)) == 0
    # statut inchangé, motifs laissés en place
    assert exec_sql(engine, "SELECT status FROM images WHERE id = %s",
                    (garee,)) == "en_attente_preannotation"
    assert exec_sql(engine, "SELECT preannotation_error_kind FROM images"
                    " WHERE id = %s", (garee,)) == "fichier_illisible"
    # de retour dans la file (avec en_file), plus comptée garée
    corps = admin.get("/api/preannotation/etat").json()
    assert corps["en_attente"] == 2 and corps["garees"] == []
