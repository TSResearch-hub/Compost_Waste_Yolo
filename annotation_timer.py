"""
Chronométrage des annotations.

Mesure le temps passé sur chaque image et journalise chaque annotation
(sauvegardée ou annulée) dans un CSV global (dataset_recolte/annotation_times.csv).
Une session (un lancement de l'app dans le navigateur) reçoit un identifiant
unique ; toutes les lignes qu'elle produit partagent ce `session_id`, ce qui
permet de filtrer/grouper les sessions entre elles dans le même fichier.
"""
import csv
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

CSV_PATH = Path("dataset_recolte") / "annotation_times.csv"

_FIELDNAMES = [
    "session_id",
    "session_date",
    "image_num",
    "image_name",
    "source",
    "start_time",
    "end_time",
    "duration_sec",
    "duration_display",
    "nb_boxes",
    "status",
]


def _ensure_session():
    """Initialise l'identifiant de session (une seule fois par session Streamlit)."""
    if "_session_id" not in st.session_state:
        now = datetime.now()
        st.session_state["_session_id"] = now.strftime("%Y%m%d_%H%M%S")
        st.session_state["_session_date"] = now.strftime("%Y-%m-%d %H:%M:%S")
        st.session_state["_session_image_count"] = 0


def start_timer(state_key: str) -> None:
    """Démarre (ou redémarre) le chronomètre pour une image donnée."""
    _ensure_session()
    st.session_state[f"_timer_start_{state_key}"] = time.time()


def get_start_time_ms(state_key: str) -> float:
    """
    Horodatage de départ du chrono en cours (epoch ms), destiné au composant
    frontend pour afficher un timer "live" (delta calculé côté JS).
    Démarre le chrono si jamais il ne l'était pas encore (sécurité).
    """
    start_ts = st.session_state.get(f"_timer_start_{state_key}")
    if start_ts is None:
        start_timer(state_key)
        start_ts = st.session_state[f"_timer_start_{state_key}"]
    return start_ts * 1000


def _format_duration(seconds: float) -> str:
    minutes, secs = divmod(max(0, int(round(seconds))), 60)
    return f"{minutes}:{secs:02d}"


def log_annotation(state_key: str, image_name: str, source: str, nb_boxes: int = 0, status: str = "completed") -> dict:
    """
    Calcule la durée écoulée depuis le dernier start_timer(state_key) et
    ajoute une ligne au CSV global. Le fichier et son en-tête sont créés
    au besoin. Retourne la ligne journalisée (pratique pour un toast).
    """
    _ensure_session()
    start_ts = st.session_state.pop(f"_timer_start_{state_key}", None)
    if start_ts is None:
        start_ts = time.time()

    end_ts = time.time()
    duration = max(0.0, end_ts - start_ts)

    st.session_state["_session_image_count"] += 1

    row = {
        "session_id": st.session_state["_session_id"],
        "session_date": st.session_state["_session_date"],
        "image_num": st.session_state["_session_image_count"],
        "image_name": image_name or "",
        "source": source,
        "start_time": datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.fromtimestamp(end_ts).strftime("%Y-%m-%d %H:%M:%S"),
        "duration_sec": round(duration, 2),
        "duration_display": _format_duration(duration),
        "nb_boxes": nb_boxes,
        "status": status,
    }

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return row
