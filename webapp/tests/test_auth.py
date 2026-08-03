"""Authentification : connexion, session cookie, expiration par inactivité."""
from conftest import exec_sql


def _login(client, username, password="motdepasse123"):
    return client.post(
        "/api/auth/login", json={"username": username, "password": password}
    )


def test_login_et_session(client, make_user):
    make_user("alice")
    r = _login(client, "alice")
    assert r.status_code == 200
    assert r.json()["username"] == "alice"
    assert "compost_session" in r.cookies

    me = client.get("/api/auth/me")
    assert me.status_code == 200
    assert me.json()["role"] == "annotateur"


def test_login_refus_message_unique(client, make_user):
    make_user("alice")
    make_user("dormant", active=False)
    cas = [
        ("alice", "mauvais-mot-de-passe"),
        ("inconnu", "motdepasse123"),
        ("dormant", "motdepasse123"),  # compte désactivé
    ]
    for username, password in cas:
        r = _login(client, username, password)
        assert r.status_code == 401
        # même message dans tous les cas : pas d'énumération de comptes
        assert r.json()["detail"] == "Identifiants invalides"


def test_sans_cookie_401(client):
    assert client.get("/api/auth/me").status_code == 401


def test_logout(client, make_user, engine):
    make_user("alice")
    _login(client, "alice")
    assert client.post("/api/auth/logout").status_code == 204
    assert client.get("/api/auth/me").status_code == 401
    # la ligne de session a bien été supprimée, pas juste le cookie
    assert exec_sql(engine, "SELECT count(*) FROM auth_sessions") == 0


def test_expiration_par_inactivite(client, make_user, engine):
    make_user("alice")
    _login(client, "alice")
    exec_sql(engine,
             "UPDATE auth_sessions SET last_seen_at = now() - interval '31 minutes'")
    assert client.get("/api/auth/me").status_code == 401
    # la session expirée a été purgée
    assert exec_sql(engine, "SELECT count(*) FROM auth_sessions") == 0


def test_renouvellement_glissant(client, make_user, engine):
    make_user("alice")
    _login(client, "alice")
    exec_sql(engine,
             "UPDATE auth_sessions SET last_seen_at = now() - interval '29 minutes'")
    assert client.get("/api/auth/me").status_code == 200
    age = exec_sql(engine,
                   "SELECT extract(epoch FROM now() - last_seen_at) FROM auth_sessions")
    assert age < 60  # l'activité a bien renouvelé la session


def test_purge_opportuniste_au_login(client, make_user, engine):
    user = make_user("alice")
    exec_sql(engine,
             "INSERT INTO auth_sessions (token_hash, user_id, last_seen_at)"
             " VALUES ('vieille-empreinte', %s, now() - interval '2 days')"
             " RETURNING token_hash", (user.id,))
    _login(client, "alice")
    restantes = exec_sql(
        engine,
        "SELECT count(*) FROM auth_sessions WHERE token_hash = 'vieille-empreinte'")
    assert restantes == 0
