"""CLI de création du premier administrateur."""
import pytest

from app import cli
from conftest import exec_sql


def test_create_admin_puis_login(monkeypatch, engine, make_client, capsys):
    monkeypatch.setenv("ADMIN_PW", "motdepasse123")
    cli.main(["create-admin", "--username", "reda",
              "--display-name", "Réda", "--password-env", "ADMIN_PW"])
    assert "Administrateur créé : reda" in capsys.readouterr().out
    assert exec_sql(
        engine,
        "SELECT role FROM users WHERE username = 'reda'") == "administrateur"

    client = make_client()
    r = client.post("/api/auth/login",
                    json={"username": "reda", "password": "motdepasse123"})
    assert r.status_code == 200


def test_refus_si_admin_actif_existe(monkeypatch, make_user):
    make_user("root", role="administrateur")
    monkeypatch.setenv("ADMIN_PW", "motdepasse123")
    with pytest.raises(SystemExit) as ei:
        cli.main(["create-admin", "--username", "reda", "--password-env", "ADMIN_PW"])
    assert "existe déjà" in str(ei.value)


def test_accepte_si_seul_admin_desactive(monkeypatch, make_user, engine):
    """Un admin désactivé ne bloque pas la création : sinon, verrouillage."""
    make_user("ancien", role="administrateur", active=False)
    monkeypatch.setenv("ADMIN_PW", "motdepasse123")
    cli.main(["create-admin", "--username", "reda", "--password-env", "ADMIN_PW"])
    assert exec_sql(
        engine,
        "SELECT count(*) FROM users WHERE role = 'administrateur'") == 2


def test_mot_de_passe_trop_court(monkeypatch):
    monkeypatch.setenv("ADMIN_PW", "court")
    with pytest.raises(SystemExit) as ei:
        cli.main(["create-admin", "--username", "reda", "--password-env", "ADMIN_PW"])
    assert "trop court" in str(ei.value)
