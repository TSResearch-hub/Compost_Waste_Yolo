"""Tests de bout en bout de l'API du serveur mobile (FastAPI TestClient).

Le module server est importé tel quel (modèle YOLO réellement chargé) mais
tous les chemins d'écriture (dataset, CSV, cache miniatures) sont redirigés
vers un répertoire temporaire : aucun test ne touche dataset_recolte/.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "mobile") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "mobile"))

import server  # noqa: E402  (charge le modèle une fois pour toute la session)


@pytest.fixture(autouse=True)
def _dataset_temporaire(tmp_path, monkeypatch):
    """Redirige toutes les écritures du serveur vers tmp_path."""
    monkeypatch.setattr(server, "SAVE_DIR", tmp_path)
    monkeypatch.setattr(server, "IMG_DIR", tmp_path / "images")
    monkeypatch.setattr(server, "LBL_DIR", tmp_path / "labels")
    monkeypatch.setattr(server, "CSV_PATH", tmp_path / "annotation_times.csv")
    monkeypatch.setattr(server, "THUMB_DIR", tmp_path / ".thumb_cache")


@pytest.fixture
def client():
    return TestClient(server.app)


def _jpeg_bytes(w=80, h=60, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _save(client, img_bytes, boxes, name="photo.jpg", overwrite=False):
    return client.post(
        "/api/save",
        files={"file": (name, img_bytes, "image/jpeg")},
        data={
            "annotations": json.dumps(boxes),
            "name": name,
            "session_id": "test",
            "duration_sec": "3.5",
            "overwrite": str(overwrite).lower(),
        },
    )


# ── config / predict ─────────────────────────────────────────────────────────

def test_config_expose_le_referentiel_complet(client):
    data = client.get("/api/config").json()
    assert len(data["classes"]) == 9
    assert "Métal" in data["classes"] and "Céramique" in data["classes"]
    assert set(data["colors"]) == set(data["classes"])


def test_predict_retourne_dimensions_et_boxes(client):
    r = client.post("/api/predict", files={"file": ("x.jpg", _jpeg_bytes(), "image/jpeg")},
                    data={"conf": "0.4"})
    assert r.status_code == 200
    data = r.json()
    assert data["width"] == 80 and data["height"] == 60
    assert isinstance(data["boxes"], list)


def test_predict_image_illisible(client):
    r = client.post("/api/predict", files={"file": ("x.jpg", b"pas une image", "image/jpeg")})
    assert r.status_code == 400


# ── save ─────────────────────────────────────────────────────────────────────

def test_save_ecrit_image_et_label_yolo(client):
    img_bytes = _jpeg_bytes()
    r = _save(client, img_bytes, [{"bbox": [20, 15, 40, 30], "label_id": 1}])
    assert r.status_code == 200
    assert r.json()["saved_name"] == "photo.jpg"

    assert (server.IMG_DIR / "photo.jpg").read_bytes() == img_bytes
    line = (server.LBL_DIR / "photo.txt").read_text().strip()
    cid, xc, yc, w, h = line.split()
    assert cid == "1"
    assert (float(xc), float(yc)) == (0.5, 0.5)   # (20+40/2)/80, (15+30/2)/60
    assert (float(w), float(h)) == (0.5, 0.5)

    csv_text = server.CSV_PATH.read_text()
    assert "mobile" in csv_text and "photo.jpg" in csv_text


def test_save_label_id_invalide_aucune_image_orpheline(client):
    r = _save(client, _jpeg_bytes(), [{"bbox": [1, 1, 5, 5], "label_id": 99}])
    assert r.status_code == 400
    # Rien ne doit avoir été écrit (bug historique : image écrite avant validation)
    assert not server.IMG_DIR.exists() or not list(server.IMG_DIR.iterdir())


def test_save_json_malforme(client):
    r = client.post(
        "/api/save",
        files={"file": ("x.jpg", _jpeg_bytes(), "image/jpeg")},
        data={"annotations": "{pas du json", "name": "x.jpg"},
    )
    assert r.status_code == 400


def test_collision_meme_contenu_reannote_en_place(client):
    img_bytes = _jpeg_bytes(seed=1)
    assert _save(client, img_bytes, [{"bbox": [1, 1, 10, 10], "label_id": 0}]).json()["saved_name"] == "photo.jpg"
    assert _save(client, img_bytes, [{"bbox": [2, 2, 12, 12], "label_id": 1}]).json()["saved_name"] == "photo.jpg"
    assert len(list(server.IMG_DIR.iterdir())) == 1


def test_collision_contenu_different_suffixe(client):
    _save(client, _jpeg_bytes(seed=1), [{"bbox": [1, 1, 10, 10], "label_id": 0}])
    r = _save(client, _jpeg_bytes(seed=2), [{"bbox": [1, 1, 10, 10], "label_id": 0}])
    assert r.json()["saved_name"] == "photo_2.jpg"


# ── gallery / image / label ──────────────────────────────────────────────────

def test_gallery_et_label(client):
    _save(client, _jpeg_bytes(), [{"bbox": [20, 15, 40, 30], "label_id": 4}])
    imgs = client.get("/api/gallery").json()["images"]
    assert [i["name"] for i in imgs] == ["photo.jpg"]
    assert imgs[0]["annotated"] is True

    lbl = client.get("/api/label/photo.jpg").json()
    assert lbl["exists"] is True
    assert lbl["boxes"][0]["label"] == "Céramique"
    bb = lbl["boxes"][0]["bbox"]
    assert pytest.approx(bb, abs=0.51) == [20, 15, 40, 30]  # aller-retour normalisation


def test_traversee_de_chemin_refusee(client):
    r = client.get("/api/image/..%2F..%2Fsettings.py")
    assert r.status_code in (400, 404)
    r = client.get("/api/thumb/.cachee.jpg")
    assert r.status_code in (400, 404)
