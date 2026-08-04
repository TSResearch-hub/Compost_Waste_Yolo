"""Sessions de capture — lecture pour l'écran d'administration technique
(rattachement d'un import, sélection d'un export). Administrateur uniquement,
comme les gestes qui s'en servent."""
from datetime import date

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func, select

from ..deps import get_db, require_roles
from ..models import CaptureSession, Image

router = APIRouter(
    prefix="/api/sessions",
    tags=["sessions"],
    dependencies=[Depends(require_roles("administrateur"))],
)


class SessionOut(BaseModel):
    id: int
    name: str
    captured_on: date
    images: int  # lignes actives


@router.get("", response_model=list[SessionOut])
def list_sessions(db=Depends(get_db)):
    comptes = (
        select(Image.session_id, func.count().label("n"))
        .where(Image.superseded_at.is_(None))
        .group_by(Image.session_id)
        .subquery()
    )
    lignes = db.execute(
        select(CaptureSession.id, CaptureSession.name,
               CaptureSession.captured_on, func.coalesce(comptes.c.n, 0))
        .outerjoin(comptes, comptes.c.session_id == CaptureSession.id)
        .order_by(CaptureSession.id)
    ).all()
    return [SessionOut(id=l[0], name=l[1], captured_on=l[2], images=l[3])
            for l in lignes]
