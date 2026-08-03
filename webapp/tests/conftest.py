"""Infrastructure de test : PostgreSQL réel obligatoire.

Résolution de la base, dans l'ordre :
1. TEST_DATABASE_URL (instance hébergée de dev — données de test uniquement) ;
2. binaires portables webapp/.pg16 (recette dans le README) : un cluster
   jetable est créé dans le tmp de pytest et détruit en fin de session.

Chaque test est isolé par TRUNCATE de toutes les tables.
"""
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

WEBAPP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEBAPP_DIR))

# Variables requises par app.config — posées avant tout import applicatif.
# Le stockage de test est un répertoire jetable, vidé entre chaque test.
STORAGE_TEST_ROOT = Path(tempfile.mkdtemp(prefix="compost_webapp_storage_"))
os.environ["STORAGE_ROOT"] = str(STORAGE_TEST_ROOT)
os.environ.setdefault(
    "DATA_YAML_PATH", str(WEBAPP_DIR.parent / "compost-yolo" / "configs" / "data.yaml")
)
os.environ.setdefault("AUTH_INACTIVITY_MINUTES", "30")

TABLES = (
    "users, auth_sessions, sessions, batches, batch_assignments, "
    "images, annotations, image_status_events"
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _spawn_pg(tmpdir: Path):
    bin_dir = Path(os.environ.get("PG16_BIN", WEBAPP_DIR / ".pg16" / "bin"))
    initdb = bin_dir / "initdb"
    if not initdb.exists():
        pytest.exit(
            "PostgreSQL introuvable : définir TEST_DATABASE_URL, ou installer "
            "les binaires portables dans webapp/.pg16 (recette dans le README).",
            returncode=3,
        )
    datadir = tmpdir / "pgdata"
    subprocess.run(
        [initdb, "-D", datadir, "-U", "compost_test", "-A", "trust",
         "-E", "UTF8", "--no-sync"],
        check=True, capture_output=True,
    )
    port = _free_port()
    subprocess.run(
        [bin_dir / "pg_ctl", "-D", datadir, "-l", tmpdir / "pg.log",
         "-o", f"-p {port} -c listen_addresses=127.0.0.1 "
               "-c unix_socket_directories='/tmp' -c fsync=off",
         "start"],
        check=True, capture_output=True,
    )
    import psycopg

    deadline = time.monotonic() + 15
    while True:
        try:
            psycopg.connect(
                f"host=127.0.0.1 port={port} user=compost_test dbname=postgres",
                connect_timeout=2,
            ).close()
            break
        except psycopg.OperationalError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.3)

    url = f"postgresql+psycopg://compost_test@127.0.0.1:{port}/postgres"

    def stop():
        subprocess.run([bin_dir / "pg_ctl", "-D", datadir, "stop"],
                       check=False, capture_output=True)

    return url, stop


@pytest.fixture(scope="session", autouse=True)
def _env(tmp_path_factory):
    url = os.environ.get("TEST_DATABASE_URL")
    stop = None
    if not url:
        url, stop = _spawn_pg(tmp_path_factory.mktemp("pg"))
    os.environ["DATABASE_URL"] = url

    from app.config import get_settings
    from app.db import get_engine, get_sessionmaker

    get_settings.cache_clear()
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()

    from alembic import command
    from alembic.config import Config

    cfg = Config(str(WEBAPP_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(WEBAPP_DIR / "alembic"))
    command.upgrade(cfg, "head")

    yield

    get_engine().dispose()
    if stop:
        stop()
    shutil.rmtree(STORAGE_TEST_ROOT, ignore_errors=True)


@pytest.fixture(autouse=True)
def _isolation(_env):
    yield
    from app.db import get_engine

    with get_engine().begin() as c:
        c.exec_driver_sql(f"TRUNCATE {TABLES} RESTART IDENTITY CASCADE")
    for child in STORAGE_TEST_ROOT.iterdir():
        shutil.rmtree(child) if child.is_dir() else child.unlink()


@pytest.fixture
def engine(_env):
    from app.db import get_engine

    return get_engine()


@pytest.fixture
def db(_env):
    from app.db import get_sessionmaker

    session = get_sessionmaker()()
    yield session
    session.rollback()
    session.close()


@pytest.fixture
def make_client(_env):
    from fastapi.testclient import TestClient

    from app.main import app

    clients = []

    def _make() -> "TestClient":
        client = TestClient(app)
        clients.append(client)
        return client

    yield _make
    for client in clients:
        client.close()


@pytest.fixture
def client(make_client):
    return make_client()


@pytest.fixture
def make_user(db):
    """Crée un compte directement en base (les tests d'API de création de
    compte passent, eux, par l'endpoint)."""
    from app.models import User
    from app.security import hash_password

    def _make(username, role="annotateur", password="motdepasse123", active=True):
        user = User(
            username=username,
            password_hash=hash_password(password),
            role=role,
            is_active=active,
        )
        db.add(user)
        db.commit()
        return user

    return _make


# ── Helpers SQL bruts (garanties du schéma) ──────────────────────────────────

def exec_sql(engine, sql, params=()):
    with engine.begin() as c:
        res = c.exec_driver_sql(sql, params)
        return res.scalar() if res.returns_rows else None


def exec_sql_all(engine, sql, params=()):
    with engine.begin() as c:
        return c.exec_driver_sql(sql, params).all()


def refus(engine, sqlstate, sql, params=()):
    """Le SQL doit être refusé avec exactement ce SQLSTATE."""
    from sqlalchemy.exc import DBAPIError

    with pytest.raises(DBAPIError) as ei:
        exec_sql(engine, sql, params)
    got = ei.value.orig.sqlstate
    assert got == sqlstate, f"SQLSTATE attendu {sqlstate}, reçu {got}"


@pytest.fixture
def base_ids(engine):
    """Jeu minimal : admin, deux annotateurs, deux sessions, un lot chacune."""
    ids = {}
    ids["admin"] = exec_sql(
        engine,
        "INSERT INTO users (username, password_hash, role)"
        " VALUES ('admin', 'x', 'administrateur') RETURNING id",
    )
    ids["alice"] = exec_sql(
        engine,
        "INSERT INTO users (username, password_hash, role, created_by)"
        " VALUES ('alice', 'x', 'annotateur', %s) RETURNING id",
        (ids["admin"],),
    )
    ids["bob"] = exec_sql(
        engine,
        "INSERT INTO users (username, password_hash, role, created_by)"
        " VALUES ('bob', 'x', 'annotateur_confirme', %s) RETURNING id",
        (ids["admin"],),
    )
    ids["s1"] = exec_sql(
        engine,
        "INSERT INTO sessions (name, captured_on, created_by)"
        " VALUES ('s1', '2026-07-01', %s) RETURNING id",
        (ids["admin"],),
    )
    ids["s2"] = exec_sql(
        engine,
        "INSERT INTO sessions (name, captured_on, created_by)"
        " VALUES ('s2', '2026-07-02', %s) RETURNING id",
        (ids["admin"],),
    )
    ids["b1"] = exec_sql(
        engine,
        "INSERT INTO batches (session_id, name, created_by)"
        " VALUES (%s, 'lot 1', %s) RETURNING id",
        (ids["s1"], ids["admin"]),
    )
    ids["b2"] = exec_sql(
        engine,
        "INSERT INTO batches (session_id, name, created_by)"
        " VALUES (%s, 'lot 1', %s) RETURNING id",
        (ids["s2"], ids["admin"]),
    )
    return ids


def insert_image(engine, ids, sha, path, **cols):
    fields = {
        "session_id": ids["s1"], "batch_id": ids["b1"],
        "original_filename": Path(path).name,
        "export_filename": Path(path).name,
        "source_label": Path(path).parent.name or "poste",
        "original_path": path,
        "width": 100, "height": 100, "sha256": sha, **cols,
    }
    names = ", ".join(fields)
    marks = ", ".join(["%s"] * len(fields))
    return exec_sql(
        engine,
        f"INSERT INTO images ({names}) VALUES ({marks}) RETURNING id",
        tuple(fields.values()),
    )
