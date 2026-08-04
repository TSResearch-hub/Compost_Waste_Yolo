"""Rôles : gestion des comptes réservée à l'administrateur."""


def _login(client, username, password="motdepasse123"):
    r = client.post(
        "/api/auth/login", json={"username": username, "password": password}
    )
    assert r.status_code == 200, r.text
    return r


def test_admin_cree_un_compte_et_il_fonctionne(make_client, make_user):
    make_user("root", role="administrateur")
    admin = make_client()
    _login(admin, "root")

    r = admin.post("/api/users", json={
        "username": "carol", "password": "motdepasse123",
        "display_name": "Carol", "role": "annotateur_confirme",
    })
    assert r.status_code == 201
    assert r.json()["role"] == "annotateur_confirme"

    carol = make_client()
    _login(carol, "carol")
    assert carol.get("/api/auth/me").json()["display_name"] == "Carol"


def test_username_deja_pris_meme_avec_casse(make_client, make_user):
    make_user("root", role="administrateur")
    make_user("alice")
    admin = make_client()
    _login(admin, "root")
    r = admin.post("/api/users", json={
        "username": "ALICE", "password": "motdepasse123", "role": "annotateur",
    })
    assert r.status_code == 409


def test_role_invalide_refuse(make_client, make_user):
    make_user("root", role="administrateur")
    admin = make_client()
    _login(admin, "root")
    r = admin.post("/api/users", json={
        "username": "eve", "password": "motdepasse123", "role": "superuser",
    })
    assert r.status_code == 422


def test_non_admins_exclus_de_la_gestion_des_comptes(make_client, make_user):
    make_user("alice", role="annotateur")
    make_user("bob", role="annotateur_confirme")
    for username in ("alice", "bob"):
        c = make_client()
        _login(c, username)
        assert c.get("/api/users").status_code == 403
        assert c.post("/api/users", json={
            "username": "x", "password": "motdepasse123", "role": "annotateur",
        }).status_code == 403


def test_desactivation_coupe_la_session_en_cours(make_client, make_user):
    make_user("root", role="administrateur")
    alice = make_user("alice")
    admin, alice_client = make_client(), make_client()
    _login(admin, "root")
    _login(alice_client, "alice")
    assert alice_client.get("/api/auth/me").status_code == 200

    r = admin.patch(f"/api/users/{alice.id}", json={"is_active": False})
    assert r.status_code == 200
    # la session existante d'alice est immédiatement invalide
    assert alice_client.get("/api/auth/me").status_code == 401
    # et elle ne peut plus se connecter
    assert alice_client.post(
        "/api/auth/login",
        json={"username": "alice", "password": "motdepasse123"},
    ).status_code == 401


def test_admin_ne_se_verrouille_pas_lui_meme(make_client, make_user):
    root = make_user("root", role="administrateur")
    admin = make_client()
    _login(admin, "root")
    for patch in ({"is_active": False}, {"role": "annotateur"}):
        r = admin.patch(f"/api/users/{root.id}", json=patch)
        assert r.status_code == 400
    # changer son propre mot de passe reste permis — et il vient de le
    # choisir : pas de changement obligatoire à la connexion suivante
    r = admin.patch(f"/api/users/{root.id}", json={"password": "nouveaumdp123"})
    assert r.status_code == 200 and r.json()["must_change_password"] is False
    _login(make_client(), "root", "nouveaumdp123")


def test_reset_mot_de_passe_d_autrui(make_client, make_user):
    make_user("root", role="administrateur")
    alice = make_user("alice")
    admin = make_client()
    _login(admin, "root")
    r = admin.patch(f"/api/users/{alice.id}", json={"password": "toutneuf12345"})
    # mot de passe posé pour autrui : à remplacer à la prochaine connexion
    assert r.status_code == 200 and r.json()["must_change_password"] is True
    c = make_client()
    assert c.post("/api/auth/login", json={
        "username": "alice", "password": "motdepasse123"}).status_code == 401
    assert _login(c, "alice", "toutneuf12345").json()[
        "must_change_password"] is True


def test_mot_de_passe_initial_doit_changer(make_client, make_user):
    """Un compte créé par l'admin doit remplacer son mot de passe initial :
    l'actuel est exigé, le changement lève l'obligation et révoque les
    AUTRES sessions du compte (un mot de passe initial a pu circuler)."""
    make_user("root", role="administrateur")
    admin = make_client()
    _login(admin, "root")
    r = admin.post("/api/users", json={
        "username": "dora", "password": "motdepasse123", "role": "annotateur"})
    assert r.status_code == 201 and r.json()["must_change_password"] is True

    dora, autre = make_client(), make_client()
    assert _login(dora, "dora").json()["must_change_password"] is True
    _login(autre, "dora")
    # mauvais mot de passe actuel : refus (un poste laissé déverrouillé ne
    # suffit pas à voler le compte)
    assert dora.post("/api/auth/changer-mot-de-passe", json={
        "actuel": "mauvais", "nouveau": "monchoix12345"}).status_code == 400
    # nouveau trop court : refus du schéma
    assert dora.post("/api/auth/changer-mot-de-passe", json={
        "actuel": "motdepasse123", "nouveau": "court"}).status_code == 422
    r = dora.post("/api/auth/changer-mot-de-passe", json={
        "actuel": "motdepasse123", "nouveau": "monchoix12345"})
    assert r.status_code == 200 and r.json()["must_change_password"] is False
    # la session courante survit, l'autre est révoquée
    assert dora.get("/api/auth/me").status_code == 200
    assert autre.get("/api/auth/me").status_code == 401
    _login(make_client(), "dora", "monchoix12345")


def test_modification_nom_affiche(make_client, make_user):
    """display_name modifiable par l'admin ; null EXPLICITE = effacer, corps
    sans le champ = intact."""
    make_user("root", role="administrateur")
    alice = make_user("alice")
    admin = make_client()
    _login(admin, "root")
    r = admin.patch(f"/api/users/{alice.id}", json={"display_name": "Alice A."})
    assert r.status_code == 200 and r.json()["display_name"] == "Alice A."
    r = admin.patch(f"/api/users/{alice.id}",
                    json={"role": "annotateur_confirme"})
    assert r.json()["display_name"] == "Alice A."
    r = admin.patch(f"/api/users/{alice.id}", json={"display_name": None})
    assert r.json()["display_name"] is None
