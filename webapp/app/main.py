"""Application FastAPI du socle."""
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .routers import (auth, batches, dataset, exports, images, imports,
                      jetsons, models_ia, preannotation, sessions, sync, users)

# Build du front (webapp/frontend, `npm run build`) — servi par le même
# process, même origine : le cookie de session suffit, pas de CORS.
FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"


def create_app() -> FastAPI:
    app = FastAPI(title="Compost — annotation multi-comptes")
    app.include_router(auth.router)
    app.include_router(users.router)
    app.include_router(imports.router)
    app.include_router(sessions.router)
    app.include_router(batches.router)
    app.include_router(images.router)
    app.include_router(exports.router)
    app.include_router(preannotation.router)
    # Flotte Jetson : registre des cartes (admin) et réception des envois
    # automatiques (jeton partagé, pas de cookie — voir routers/sync.py)
    app.include_router(jetsons.router)
    app.include_router(sync.router)
    # Distribution des poids du modèle IA : publication par un administrateur,
    # récupération par les cartes avec le même jeton (voir routers/models_ia.py)
    app.include_router(models_ia.router)
    # Écran Dataset : export YOLO en ZIP à la volée et dépôt d'images brutes
    # depuis le PC de l'administrateur (voir routers/dataset.py)
    app.include_router(dataset.router)

    @app.get("/api/health", tags=["meta"])
    def health():
        return {"status": "ok"}

    # Monté en dernier : les routes /api déclarées avant restent prioritaires.
    # Absent (front non construit), l'API fonctionne seule — rien ne casse.
    if FRONTEND_DIST.is_dir():
        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True),
                  name="frontend")

    return app


app = create_app()
