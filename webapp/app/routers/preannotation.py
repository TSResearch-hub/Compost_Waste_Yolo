"""File de pré-annotation — observation et relance, administrateur uniquement.

Le worker n'est PAS piloté d'ici : il tourne à part (machine dédiée à terme)
et ne communique que par la base. Cet écran observe sa file et peut remettre
une image « garée » en circulation en remettant son compteur de tentatives à
zéro — sans toucher au statut (aucune transition, la liste blanche CW002
n'est pas concernée). Les motifs d'échec sont laissés en place : ils
documentent le dernier échec jusqu'à ce que le worker les efface au succès.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from ..config import get_settings
from ..deps import get_db, require_roles
from ..models import Batch, CaptureSession, Image

router = APIRouter(
    prefix="/api/preannotation",
    tags=["preannotation"],
    dependencies=[Depends(require_roles("administrateur"))],
)


class GareeOut(BaseModel):
    id: int
    nom: str  # export_filename
    lot: str
    session: str
    tentatives: int
    motif_type: str | None  # PREANNOTATION_ERROR_KINDS
    motif: str | None


class EtatFileOut(BaseModel):
    # encore dans la file : le worker les prendra (tentatives sous le plafond)
    en_attente: int
    plafond_tentatives: int
    garees: list[GareeOut]


class RelancerIn(BaseModel):
    image_ids: list[int] = Field(min_length=1)


class RelancerOut(BaseModel):
    relancees: int


@router.get("/etat", response_model=EtatFileOut)
def etat_file(db=Depends(get_db)):
    plafond = get_settings().worker_max_attempts
    en_attente = db.scalar(
        select(func.count()).select_from(Image)
        .where(Image.status == "en_attente_preannotation",
               Image.superseded_at.is_(None),
               Image.preannotation_attempts < plafond)
    )
    lignes = db.execute(
        select(Image.id, Image.export_filename, Batch.name,
               CaptureSession.name, Image.preannotation_attempts,
               Image.preannotation_error_kind, Image.preannotation_error)
        .join(Batch, Batch.id == Image.batch_id)
        .join(CaptureSession, CaptureSession.id == Image.session_id)
        .where(Image.status == "en_attente_preannotation",
               Image.superseded_at.is_(None),
               Image.preannotation_attempts >= plafond)
        .order_by(Image.id)
    ).all()
    return EtatFileOut(
        en_attente=en_attente, plafond_tentatives=plafond,
        garees=[GareeOut(id=l[0], nom=l[1], lot=l[2], session=l[3],
                         tentatives=l[4], motif_type=l[5], motif=l[6])
                for l in lignes],
    )


@router.post("/relancer", response_model=RelancerOut)
def relancer(body: RelancerIn, db=Depends(get_db)):
    """Remet le compteur de tentatives à zéro : l'image repasse dans la file.
    Tout-ou-rien, comme le court-circuit : une image inconnue/archivée (404)
    ou hors file (409) fait échouer le lot entier, rien n'est écrit."""
    ids = set(body.image_ids)
    actives = {
        i.id: i for i in db.scalars(select(Image).where(Image.id.in_(ids)))
        if i.superseded_at is None
    }
    inconnues = sorted(ids - actives.keys())
    if inconnues:
        raise HTTPException(
            status_code=404,
            detail=f"Images inconnues ou archivées : {inconnues}")
    hors_file = sorted(
        i.id for i in actives.values()
        if i.status != "en_attente_preannotation"
    )
    if hors_file:
        raise HTTPException(
            status_code=409,
            detail="Images hors file de pré-annotation "
                   f"(rien n'a été modifié) : {hors_file}")
    for image in actives.values():
        image.preannotation_attempts = 0
    return RelancerOut(relancees=len(actives))
