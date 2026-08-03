"""Application FastAPI du socle."""
from fastapi import FastAPI

from .routers import auth, batches, exports, images, imports, users


def create_app() -> FastAPI:
    app = FastAPI(title="Compost — annotation multi-comptes")
    app.include_router(auth.router)
    app.include_router(users.router)
    app.include_router(imports.router)
    app.include_router(batches.router)
    app.include_router(images.router)
    app.include_router(exports.router)

    @app.get("/api/health", tags=["meta"])
    def health():
        return {"status": "ok"}

    return app


app = create_app()
