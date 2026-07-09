"""
Serveur de la version mobile (PWA) de l'outil d'annotation.

Rôle : le téléphone/la tablette n'est qu'un écran tactile — toute
l'intelligence (modèle YOLO) et le stockage (dataset_recolte/) restent sur
le PC. Le mobile s'y connecte en WiFi via l'IP locale du PC.

    ┌────────────┐   photo (multipart)   ┌──────────────────────────┐
    │  Téléphone  │ ────────────────────▶ │  PC : FastAPI + YOLO     │
    │  (PWA)      │ ◀──────────────────── │  → dataset_recolte/      │
    └────────────┘   bboxes pré-annotées  └──────────────────────────┘

Lancement (depuis la racine du projet ou n'importe où) :
    venv/bin/python mobile/server.py            # port 8000 par défaut
    venv/bin/python mobile/server.py --port 9000

Puis sur le mobile (même réseau WiFi) : http://<IP-du-PC>:8000
L'IP est affichée au démarrage.
"""
import argparse
import csv
import json
import socket
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ── Chemins absolus : indépendants du répertoire de lancement ────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR   = Path(__file__).resolve().parent / "static"
SAVE_DIR     = PROJECT_ROOT / "dataset_recolte"
IMG_DIR      = SAVE_DIR / "images"
LBL_DIR      = SAVE_DIR / "labels"
CSV_PATH     = SAVE_DIR / "annotation_times.csv"
MODEL_PATH   = PROJECT_ROOT / "weights" / "best.pt"

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}

sys.path.insert(0, str(PROJECT_ROOT))
import helper  # noqa: E402  (CLASS_MAP, CLASS_COLORS, get_detection_initial_data)

# Mêmes colonnes que annotation_timer.py → un seul CSV pour PC et mobile
CSV_FIELDNAMES = [
    "session_id", "session_date", "image_num", "image_name", "source",
    "start_time", "end_time", "duration_sec", "duration_display",
    "nb_boxes", "status",
]

app = FastAPI(title="Composte IA — annotation mobile")

# Modèle chargé une seule fois au démarrage (~2-5 s)
model = helper.load_model(MODEL_PATH)

_csv_lock = threading.Lock()
_session_counters: dict[str, int] = {}


# ══════════════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ══════════════════════════════════════════════════════════════════════════════

def _decode_upload(data: bytes):
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Image illisible")
    return img


def _resolve_name_collision(stem: str, ext: str, raw_bytes: bytes) -> str:
    """Même logique que l'app PC : ré-annotation du même contenu → même nom ;
    collision de nom avec un contenu différent → suffixe _2, _3, ..."""
    candidate, i = stem, 1
    while (IMG_DIR / f"{candidate}{ext}").exists():
        if (IMG_DIR / f"{candidate}{ext}").read_bytes() == raw_bytes:
            return candidate
        i += 1
        candidate = f"{stem}_{i}"
    return candidate


def _format_duration(seconds: float) -> str:
    minutes, secs = divmod(max(0, int(round(seconds))), 60)
    return f"{minutes}:{secs:02d}"


def _log_annotation(session_id: str, image_name: str, duration_sec: float, nb_boxes: int):
    """Ajoute une ligne au CSV commun de chronométrage (source = mobile)."""
    with _csv_lock:
        _session_counters[session_id] = _session_counters.get(session_id, 0) + 1
        end_ts = time.time()
        start_ts = end_ts - max(0.0, duration_sec)
        row = {
            "session_id": session_id,
            "session_date": datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d %H:%M:%S"),
            "image_num": _session_counters[session_id],
            "image_name": image_name,
            "source": "mobile",
            "start_time": datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": datetime.fromtimestamp(end_ts).strftime("%Y-%m-%d %H:%M:%S"),
            "duration_sec": round(duration_sec, 2),
            "duration_display": _format_duration(duration_sec),
            "nb_boxes": nb_boxes,
            "status": "completed",
        }
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        write_header = not CSV_PATH.exists()
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def _safe_image_path(name: str) -> Path:
    """Résout un nom d'image du dataset en refusant toute traversée de chemin."""
    if "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(status_code=400, detail="Nom de fichier invalide")
    path = IMG_DIR / name
    if not path.exists() or path.suffix.lower() not in IMG_EXTENSIONS:
        raise HTTPException(status_code=404, detail="Image introuvable")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# API
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/config")
def get_config():
    """Classes, couleurs et seuil par défaut — source unique : helper.py."""
    return {
        "classes": list(helper.CLASS_MAP.keys()),
        "colors": helper.CLASS_COLORS,
        "conf_default": 0.4,
    }


@app.post("/api/predict")
def predict(file: UploadFile = File(...), conf: float = Form(0.4)):
    """Pré-annotation IA : image → bboxes [x, y, w, h] en pixels + classes."""
    img = _decode_upload(file.file.read())
    h, w = img.shape[:2]
    results = model.predict(img, conf=float(conf))
    bboxes, labels = helper.get_detection_initial_data(results[0])
    class_names = list(helper.CLASS_MAP.keys())
    return {
        "width": w,
        "height": h,
        "boxes": [
            {"bbox": [round(v, 2) for v in bb], "label_id": lid, "label": class_names[lid]}
            for bb, lid in zip(bboxes, labels)
        ],
    }


@app.post("/api/save")
def save(
    file: UploadFile = File(...),
    annotations: str = Form(...),      # JSON : [{"bbox":[x,y,w,h], "label_id":int}, ...]
    name: str = Form(""),
    session_id: str = Form("mobile"),
    duration_sec: float = Form(0.0),
    overwrite: bool = Form(False),     # True en mode vérification (correction en place)
):
    """Sauvegarde image + label YOLO dans dataset_recolte/ (format identique au PC)."""
    raw_bytes = file.file.read()
    img = _decode_upload(raw_bytes)
    h_img, w_img = img.shape[:2]

    try:
        boxes = json.loads(annotations)
        assert isinstance(boxes, list)
    except (json.JSONDecodeError, AssertionError):
        raise HTTPException(status_code=400, detail="Annotations JSON invalides")

    # Validation COMPLÈTE avant toute écriture : ne jamais laisser une image
    # orpheline (sans label) dans le dataset si les annotations sont invalides.
    n_classes = len(helper.CLASS_MAP)
    yolo_lines = []
    try:
        for item in boxes:
            x, y, w_box, h_box = (float(v) for v in item["bbox"])
            label_id = int(item["label_id"])
            if not 0 <= label_id < n_classes:
                raise HTTPException(status_code=400, detail=f"label_id invalide : {label_id}")
            yolo_lines.append(
                f"{label_id} "
                f"{(x + w_box / 2) / w_img:.6f} {(y + h_box / 2) / h_img:.6f} "
                f"{w_box / w_img:.6f} {h_box / h_img:.6f}"
            )
    except (KeyError, TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Format d'annotation invalide")

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    LBL_DIR.mkdir(parents=True, exist_ok=True)

    if name:
        stem = Path(name).stem
        ext = Path(name).suffix.lower() or ".jpg"
    else:
        stem = f"mob_{int(time.time())}"
        ext = ".jpg"
    if not overwrite:
        stem = _resolve_name_collision(stem, ext, raw_bytes)

    (IMG_DIR / f"{stem}{ext}").write_bytes(raw_bytes)
    (LBL_DIR / f"{stem}.txt").write_text("\n".join(yolo_lines), encoding="utf-8")

    _log_annotation(session_id, f"{stem}{ext}", duration_sec, len(boxes))
    return {"saved_name": f"{stem}{ext}", "nb_boxes": len(boxes)}


@app.get("/api/gallery")
def gallery():
    """Liste du dataset : nom + statut annoté, du plus récent au plus ancien."""
    if not IMG_DIR.exists():
        return {"images": []}
    items = sorted(
        (p for p in IMG_DIR.iterdir() if p.suffix.lower() in IMG_EXTENSIONS),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return {
        "images": [
            {"name": p.name, "annotated": (LBL_DIR / f"{p.stem}.txt").exists()}
            for p in items
        ]
    }


@app.get("/api/image/{name}")
def get_image(name: str):
    return FileResponse(_safe_image_path(name))


THUMB_DIR = Path(__file__).resolve().parent / ".thumb_cache"
THUMB_SIZE = 256


@app.get("/api/thumb/{name}")
def get_thumb(name: str):
    """
    Miniature 256px pour la galerie. Les photos du dataset font plusieurs Mo
    (3000×4000) : servir les originaux rendrait la galerie inutilisable sur
    mobile. Générée à la volée puis cachée sur disque (invalidée si l'image
    source change de mtime).
    """
    src = _safe_image_path(name)
    THUMB_DIR.mkdir(exist_ok=True)
    thumb_path = THUMB_DIR / f"{src.stem}_{int(src.stat().st_mtime)}.jpg"
    if not thumb_path.exists():
        img = cv2.imread(str(src))
        if img is None:
            raise HTTPException(status_code=500, detail="Image illisible")
        h, w = img.shape[:2]
        scale = THUMB_SIZE / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(thumb_path), img, [cv2.IMWRITE_JPEG_QUALITY, 72])
    return FileResponse(thumb_path, media_type="image/jpeg")


@app.get("/api/label/{name}")
def get_label(name: str):
    """Label YOLO d'une image du dataset, converti en pixels pour l'éditeur."""
    img_path = _safe_image_path(name)
    img = cv2.imread(str(img_path))
    if img is None:
        raise HTTPException(status_code=500, detail="Image illisible")
    h_img, w_img = img.shape[:2]

    lbl_path = LBL_DIR / f"{img_path.stem}.txt"
    boxes = []
    class_names = list(helper.CLASS_MAP.keys())
    if lbl_path.exists():
        for line in lbl_path.read_text(encoding="utf-8").strip().split("\n"):
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    cid = int(parts[0])
                    xc, yc, bw, bh = (float(p) for p in parts[1:])
                except ValueError:
                    continue
                cid = min(cid, len(class_names) - 1)
                boxes.append({
                    "bbox": [(xc - bw / 2) * w_img, (yc - bh / 2) * h_img, bw * w_img, bh * h_img],
                    "label_id": cid,
                    "label": class_names[cid],
                })
    return {"width": w_img, "height": h_img, "boxes": boxes, "exists": lbl_path.exists()}


@app.get("/api/stats")
def stats():
    """Compteurs affichés sur l'écran d'accueil mobile."""
    n_images = (
        len([p for p in IMG_DIR.iterdir() if p.suffix.lower() in IMG_EXTENSIONS])
        if IMG_DIR.exists() else 0
    )
    n_labels = len(list(LBL_DIR.glob("*.txt"))) if LBL_DIR.exists() else 0
    today = datetime.now().strftime("%Y-%m-%d")
    n_today, durations = 0, []
    if CSV_PATH.exists():
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "completed" and row.get("end_time", "").startswith(today):
                    n_today += 1
                    try:
                        durations.append(float(row["duration_sec"]))
                    except (TypeError, ValueError):
                        pass
    return {
        "images": n_images,
        "annotated": n_labels,
        "today": n_today,
        "mean_sec_today": round(sum(durations) / len(durations), 1) if durations else None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FICHIERS STATIQUES (PWA)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


# Le service worker doit être servi depuis la racine pour contrôler tout le site
@app.get("/sw.js")
def service_worker():
    return FileResponse(STATIC_DIR / "sw.js", media_type="application/javascript")


@app.get("/manifest.webmanifest")
def manifest():
    return FileResponse(STATIC_DIR / "manifest.webmanifest", media_type="application/manifest+json")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _local_ip() -> str:
    """IP locale du PC sur le réseau (pour l'afficher au démarrage)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Serveur d'annotation mobile")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    print("\n" + "═" * 60)
    print("  Composte IA — annotation mobile")
    print(f"  Sur ce PC     : http://localhost:{args.port}")
    print(f"  Sur le mobile : http://{_local_ip()}:{args.port}")
    print("  (téléphone et PC doivent être sur le même réseau WiFi)")
    print("═" * 60 + "\n")

    uvicorn.run(app, host=args.host, port=args.port)
