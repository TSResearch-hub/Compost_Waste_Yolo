"""Configuration par variables d'environnement — aucune valeur en dur.

Voir .env.example pour la liste. En dev, les variables peuvent venir d'un
fichier webapp/.env (jamais commité).
"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    database_url: str
    storage_root: Path
    data_yaml_path: Path
    auth_inactivity_minutes: int = 30

    # ── Worker de pré-annotation (lot L2) ────────────────────────────────────
    # Poids YOLO : requis par le worker seul, le serveur web démarre sans
    weights_path: Path | None = None
    # Seuil volontairement bas : une boîte de trop se supprime plus vite
    # qu'une boîte manquante ne se dessine — à calibrer via `worker essai`
    worker_conf_threshold: float = 0.10
    # Plafond de détections par image : le max historique observé est de
    # 9 boîtes/image (p99 = 6) — 20 laisse plus du double de marge tout en
    # coupant les rafales de détections de texture
    worker_max_det: int = 20
    worker_poll_seconds: float = 5.0
    # Au-delà : image « garée », sortie de la file, visible dans les lots
    worker_max_attempts: int = 3
    # Échecs consécutifs avant arrêt du worker : des poids cassés sont un
    # problème de worker, pas d'images — il ne doit pas brûler la file
    worker_max_consecutive_failures: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
