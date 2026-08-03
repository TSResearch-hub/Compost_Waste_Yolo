"""Export YOLO côté API — même service que la CLI, le répertoire de sortie
est un chemin côté serveur (administrateur uniquement). L'export ne modifie
ni la base ni le stockage : seule la sortie est écrite."""
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..deps import get_db, require_roles
from ..exporter import export_yolo
from ..models import User
from ..storage import get_storage

router = APIRouter(prefix="/api/exports", tags=["exports"])


class ExportIn(BaseModel):
    output_dir: str
    # None = tout exporter
    session_names: list[str] | None = None


class SessionStatOut(BaseModel):
    id: int
    name: str
    images: int
    boxes: int


class ExportReportOut(BaseModel):
    output_dir: str
    images: int
    boxes: int
    empty_labels: int
    sessions: list[SessionStatOut]
    class_counts: dict[str, int]
    renamed: list[tuple[str, str]]


@router.post("", response_model=ExportReportOut, status_code=201)
def create_export(
    body: ExportIn,
    db=Depends(get_db),
    admin: User = Depends(require_roles("administrateur")),
):
    try:
        report = export_yolo(db, get_storage(), output_dir=Path(body.output_dir),
                             session_names=body.session_names)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return ExportReportOut(
        output_dir=report.output_dir, images=report.images, boxes=report.boxes,
        empty_labels=report.empty_labels,
        sessions=[SessionStatOut(id=s[0], name=s[1], images=s[2], boxes=s[3])
                  for s in report.sessions],
        class_counts=report.class_counts, renamed=report.renamed,
    )
