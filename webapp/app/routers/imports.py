"""Import d'un lot — administrateur uniquement.

Le dossier source est un chemin CÔTÉ SERVEUR (déposé par l'administrateur) ;
il est lu, jamais modifié. Refus (409) si rien n'est importable, avec le
rapport complet en détail.
"""
from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..deps import get_db, require_roles
from ..importer import ImportReport, import_session_folder
from ..models import User
from ..storage import get_storage

router = APIRouter(prefix="/api/imports", tags=["imports"])


class ImportIn(BaseModel):
    # Plusieurs dossiers (un par poste de capture) = UNE session : capturer le
    # même tas au même moment dans des sessions séparées créerait une fuite
    # de données au split train/test
    source_dirs: list[str] = Field(min_length=1)
    name: str = Field(min_length=1)
    captured_on: date
    lighting: str | None = None
    camera_height_cm: int | None = Field(default=None, ge=0)
    compost_state: str | None = None
    operator: str | None = None
    notes: str | None = None


class ImportReportOut(BaseModel):
    session_name: str
    session_id: int | None
    batch_id: int | None
    created: list[str]
    duplicates: list[tuple[str, str]]
    rejected: list[tuple[str, str]]
    renamed: list[tuple[str, str]]
    aborted_reason: str | None


def _to_out(report: ImportReport) -> ImportReportOut:
    return ImportReportOut(
        session_name=report.session_name, session_id=report.session_id,
        batch_id=report.batch_id, created=report.created,
        duplicates=report.duplicates, rejected=report.rejected,
        renamed=report.renamed, aborted_reason=report.aborted_reason,
    )


@router.post("", response_model=ImportReportOut, status_code=201)
def import_lot(
    body: ImportIn,
    db=Depends(get_db),
    admin: User = Depends(require_roles("administrateur")),
):
    try:
        report = import_session_folder(
            db, get_storage(),
            source_dirs=body.source_dirs, admin_id=admin.id, name=body.name,
            captured_on=body.captured_on, lighting=body.lighting,
            camera_height_cm=body.camera_height_cm,
            compost_state=body.compost_state, operator=body.operator,
            notes=body.notes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if report.aborted:
        raise HTTPException(status_code=409, detail=_to_out(report).model_dump())
    return _to_out(report)
