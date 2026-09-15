"""Distribution des poids du modèle IA : publication par un administrateur
(POST /api/models_ia/upload — formulaire, deux fichiers, contrôles, jamais
d'écrasement), lecture de la dernière version et téléchargement par les
cartes avec le jeton SYNC_TOKEN (GET /api/models_ia/latest,
GET /api/models_ia/{id}/fichier/{pt|yaml})."""
import os

import pytest

from conftest import STORAGE_TEST_ROOT, exec_sql, exec_sql_all

TOKEN = "jeton-de-test-suffisamment-long"
PT = b"PK\x03\x04poids factices du modele"
YAML = b"names:\n  - Plastique\n  - Metal\n"


@pytest.fixture
def sync_env(monkeypatch):
    """SYNC_TOKEN posé pour le test ; la config (lru_cache) est rechargée à
    l'entrée et après restauration de l'environnement."""
    from app.config import get_settings

    monkeypatch.setenv("SYNC_TOKEN", TOKEN)
    get_settings.cache_clear()
    yield monkeypatch
    monkeypatch.undo()
    get_settings.cache_clear()


def login(client, username, password="motdepasse123"):
    r = client.post("/api/auth/login",
                    json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return client


def publier(client, version_name, *, pt=PT, yaml=YAML, pt_nom="best.pt",
            yaml_nom="data.yaml"):
    return client.post(
        "/api/models_ia/upload", data={"version_name": version_name},
        files={"pt_file": (pt_nom, pt, "application/octet-stream"),
               "yaml_file": (yaml_nom, yaml, "application/yaml")})


def carte(client, path, token=TOKEN):
    headers = {"Authorization": f"Bearer {token}"} if token is not None else {}
    return client.get(path, headers=headers)


def fichiers_stockage():
    return sorted(str(p.relative_to(STORAGE_TEST_ROOT))
                  for p in STORAGE_TEST_ROOT.rglob("*") if p.is_file())


def test_publication_admin_seulement(make_client, make_user, engine):
    admin = make_user("root", role="administrateur")
    make_user("alice", role="annotateur_confirme")
    assert publier(make_client(), "v1").status_code == 401
    assert publier(login(make_client(), "alice"), "v1").status_code == 403
    assert exec_sql(engine, "SELECT count(*) FROM model_versions") == 0
    assert fichiers_stockage() == []

    r = publier(login(make_client(), "root"), "yolov8n_2026-09-15")
    assert r.status_code == 201, r.text
    corps = r.json()
    assert corps["version_name"] == "yolov8n_2026-09-15"
    assert corps["pt_file_path"] == "models/yolov8n_2026-09-15/model.pt"
    assert corps["yaml_file_path"] == "models/yolov8n_2026-09-15/data.yaml"
    assert corps["created_by"] == admin.id and corps["created_at"]
    assert corps["pt_url"] == f"/api/models_ia/{corps['id']}/fichier/pt"
    assert corps["yaml_url"] == f"/api/models_ia/{corps['id']}/fichier/yaml"
    # fichiers posés dans le stockage, sous models/, octets intacts
    assert fichiers_stockage() == ["models/yolov8n_2026-09-15/data.yaml",
                                   "models/yolov8n_2026-09-15/model.pt"]
    assert (STORAGE_TEST_ROOT / corps["pt_file_path"]).read_bytes() == PT
    assert (STORAGE_TEST_ROOT / corps["yaml_file_path"]).read_bytes() == YAML
    assert exec_sql_all(
        engine, "SELECT version_name, pt_file_path, created_by FROM model_versions"
    ) == [("yolov8n_2026-09-15", "models/yolov8n_2026-09-15/model.pt", admin.id)]


def test_entrees_refusees_sans_rien_ecrire(make_client, make_user, engine):
    make_user("root", role="administrateur")
    admin = login(make_client(), "root")
    # nom de version hors forme : vide, espace, séparateur de chemin, `..`,
    # trop long — il nomme un dossier du stockage
    for mauvais in ("", "v 1", "v/1", "../v1", "v" * 101):
        assert publier(admin, mauvais).status_code == 422, mauvais
    # extensions
    r = publier(admin, "v1", pt_nom="best.onnx")
    assert r.status_code == 422 and ".pt" in r.json()["detail"]
    r = publier(admin, "v1", yaml_nom="data.txt")
    assert r.status_code == 422 and ".yaml" in r.json()["detail"]
    # .pt vide
    r = publier(admin, "v1", pt=b"")
    assert r.status_code == 422 and "vide" in r.json()["detail"]
    # data.yaml illisible, ou sans liste `names`
    for yaml in (b"\x00\xff\xfe pas du yaml", b"names: 3\n", b"nc: 2\n", b"names: []\n"):
        r = publier(admin, "v1", yaml=yaml)
        assert r.status_code == 422 and "data.yaml" in r.json()["detail"], yaml
    # la base refuse aussi une forme non canonique (CHECK)
    from conftest import refus
    refus(engine, "23514",
          "INSERT INTO model_versions (version_name, pt_file_path, yaml_file_path,"
          " created_by) VALUES ('a b', 'x', 'y', (SELECT id FROM users))")
    assert exec_sql(engine, "SELECT count(*) FROM model_versions") == 0
    assert fichiers_stockage() == []


def test_version_unique_jamais_ecrasee(make_client, make_user, engine):
    make_user("root", role="administrateur")
    admin = login(make_client(), "root")
    assert publier(admin, "v1").status_code == 201
    r = publier(admin, "v1", pt=b"autres poids")
    assert r.status_code == 409 and "déjà publiée" in r.json()["detail"]
    assert exec_sql(engine, "SELECT count(*) FROM model_versions") == 1
    assert (STORAGE_TEST_ROOT / "models/v1/model.pt").read_bytes() == PT
    # fichiers présents sans ligne (reste d'un incident) : refus explicite,
    # le stockage ne réécrit jamais
    exec_sql(engine, "DELETE FROM model_versions")
    r = publier(admin, "v1")
    assert r.status_code == 409 and "stockage" in r.json()["detail"]
    assert exec_sql(engine, "SELECT count(*) FROM model_versions") == 0


def test_plafond_de_taille_413(make_client, make_user, engine):
    from app.config import get_settings

    make_user("root", role="administrateur")
    admin = login(make_client(), "root")
    os.environ["MODELS_MAX_UPLOAD_MB"] = "1"
    get_settings.cache_clear()
    try:
        r = publier(admin, "gros", pt=os.urandom(1_600_000))
        assert r.status_code == 413 and "MODELS_MAX_UPLOAD_MB" in r.json()["detail"]
    finally:
        del os.environ["MODELS_MAX_UPLOAD_MB"]
        get_settings.cache_clear()
    assert exec_sql(engine, "SELECT count(*) FROM model_versions") == 0
    assert fichiers_stockage() == []


def test_latest_protegee_par_le_jeton(make_client, make_user, engine,
                                      monkeypatch):
    from app.config import get_settings

    make_user("root", role="administrateur")
    admin = login(make_client(), "root")
    anonyme = make_client()
    # SYNC_TOKEN non configuré : distribution coupée, comme la réception
    monkeypatch.delenv("SYNC_TOKEN", raising=False)
    get_settings.cache_clear()
    try:
        r = carte(anonyme, "/api/models_ia/latest")
        assert r.status_code == 503 and "SYNC_TOKEN" in r.json()["detail"]
    finally:
        get_settings.cache_clear()
    monkeypatch.setenv("SYNC_TOKEN", TOKEN)
    get_settings.cache_clear()
    try:
        assert carte(anonyme, "/api/models_ia/latest", token=None).status_code == 401
        assert carte(anonyme, "/api/models_ia/latest", token="mauvais").status_code == 401
        # le cookie d'un administrateur ne remplace pas le jeton
        assert admin.get("/api/models_ia/latest").status_code == 401
        # rien de publié
        r = carte(anonyme, "/api/models_ia/latest")
        assert r.status_code == 404 and "Aucune version" in r.json()["detail"]

        assert publier(admin, "v1").status_code == 201
        assert publier(admin, "v2", pt=b"poids v2").status_code == 201
        # antidater v1 pour que l'ordre ne dépende pas de l'horloge
        exec_sql(engine, "UPDATE model_versions SET created_at = created_at"
                         " - interval '1 day' WHERE version_name = 'v1'")
        r = carte(anonyme, "/api/models_ia/latest")
        assert r.status_code == 200, r.text
        assert r.json()["version_name"] == "v2"
        assert r.json()["pt_url"].endswith("/fichier/pt")
        # une v3 publiée ensuite devient la dernière
        assert publier(admin, "v3", pt=b"poids v3").status_code == 201
        assert carte(anonyme, "/api/models_ia/latest").json()["version_name"] == "v3"
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()


def test_telechargement_des_fichiers(make_client, make_user, sync_env):
    make_user("root", role="administrateur")
    admin = login(make_client(), "root")
    anonyme = make_client()
    r = publier(admin, "v1")
    assert r.status_code == 201
    pt_url, yaml_url = r.json()["pt_url"], r.json()["yaml_url"]

    assert carte(anonyme, pt_url, token=None).status_code == 401
    assert carte(anonyme, pt_url, token="mauvais").status_code == 401
    assert admin.get(pt_url).status_code == 401  # cookie ≠ jeton

    r = carte(anonyme, pt_url)
    assert r.status_code == 200, r.text
    assert r.content == PT
    assert r.headers["content-type"].startswith("application/octet-stream")
    assert r.headers["content-length"] == str(len(PT))
    assert r.headers["content-disposition"] == 'attachment; filename="model.pt"'

    r = carte(anonyme, yaml_url)
    assert r.status_code == 200 and r.content == YAML
    assert r.headers["content-type"].startswith("application/yaml")
    assert r.headers["content-disposition"] == 'attachment; filename="data.yaml"'

    # version inconnue, sorte de fichier inconnue
    assert carte(anonyme, "/api/models_ia/999/fichier/pt").status_code == 404
    assert carte(anonyme, "/api/models_ia/1/fichier/onnx").status_code == 422
    # ligne sans fichier : incohérence serveur, pas un « pas de modèle »
    (STORAGE_TEST_ROOT / "models/v1/model.pt").unlink()
    r = carte(anonyme, pt_url)
    assert r.status_code == 500 and "absent du stockage" in r.json()["detail"]


def test_liste_administrateur(make_client, make_user, engine, sync_env):
    make_user("root", role="administrateur")
    make_user("alice")
    admin = login(make_client(), "root")
    assert make_client().get("/api/models_ia").status_code == 401
    assert login(make_client(), "alice").get("/api/models_ia").status_code == 403
    # le jeton des cartes ne donne pas accès à la liste
    assert carte(make_client(), "/api/models_ia").status_code == 401
    assert admin.get("/api/models_ia").json() == []
    assert publier(admin, "v1").status_code == 201
    assert publier(admin, "v2").status_code == 201
    exec_sql(engine, "UPDATE model_versions SET created_at = created_at"
                     " - interval '1 day' WHERE version_name = 'v1'")
    assert [v["version_name"] for v in admin.get("/api/models_ia").json()] == [
        "v2", "v1"]
