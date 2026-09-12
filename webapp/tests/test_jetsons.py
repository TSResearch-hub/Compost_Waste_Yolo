"""Registre des cartes Jetson : administrateur uniquement, identifiant
normalisé (MAC avec `:` ou majuscules → forme canonique), désactivation sans
suppression."""
from conftest import refus


def login(client, username, password="motdepasse123"):
    r = client.post("/api/auth/login",
                    json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return client


def test_admin_seulement(make_client, make_user):
    make_user("root", role="administrateur")
    make_user("alice")
    assert make_client().get("/api/jetsons").status_code == 401
    alice = login(make_client(), "alice")
    assert alice.get("/api/jetsons").status_code == 403
    assert alice.post("/api/jetsons", json={"id": "x"}).status_code == 403
    assert alice.patch("/api/jetsons/x",
                       json={"is_active": False}).status_code == 403


def test_declaration_normalisee_et_unicite(make_client, make_user, engine):
    make_user("root", role="administrateur")
    admin = login(make_client(), "root")
    r = admin.post("/api/jetsons",
                   json={"id": " 48:B0:2D:3E:AA:01 ", "name": "Tas A"})
    assert r.status_code == 201, r.text
    corps = r.json()
    assert corps["id"] == "48-b0-2d-3e-aa-01"
    assert corps["name"] == "Tas A" and corps["is_active"] is True
    assert corps["created_at"] and corps["updated_at"]
    # même carte sous une autre écriture : déjà déclarée
    assert admin.post("/api/jetsons",
                      json={"id": "48-B0-2D-3E-AA-01"}).status_code == 409
    # identifiants hors forme : vide, séparateur de chemin, espace, trop long
    for mauvais in ("", "tas/A", "a b", "x" * 101):
        assert admin.post("/api/jetsons",
                          json={"id": mauvais}).status_code == 422, mauvais
    # la base elle-même refuse une forme non canonique (CHECK)
    refus(engine, "23514", "INSERT INTO jetson_devices (id) VALUES ('AA:BB')")


def test_liste_et_modification(make_client, make_user):
    make_user("root", role="administrateur")
    admin = login(make_client(), "root")
    assert admin.post("/api/jetsons", json={"id": "b-02"}).status_code == 201
    assert admin.post("/api/jetsons",
                      json={"id": "a-01", "name": "Un"}).status_code == 201
    r = admin.get("/api/jetsons")
    assert r.status_code == 200
    assert [(c["id"], c["name"]) for c in r.json()] == [("a-01", "Un"),
                                                        ("b-02", None)]
    # chemin normalisé comme à la création ; champs absents intacts
    r = admin.patch("/api/jetsons/A:01", json={"is_active": False})
    assert r.status_code == 200, r.text
    assert r.json()["is_active"] is False and r.json()["name"] == "Un"
    # null explicite = effacer le nom ; corps sans `name` = intact
    assert admin.patch("/api/jetsons/a-01",
                       json={"name": None}).json()["name"] is None
    assert admin.patch("/api/jetsons/a-01",
                       json={"is_active": True}).json()["name"] is None
    assert admin.patch("/api/jetsons/a-01",
                       json={"name": "Deux"}).json()["name"] == "Deux"
    assert admin.patch("/api/jetsons/inconnue",
                       json={"name": "x"}).status_code == 404
