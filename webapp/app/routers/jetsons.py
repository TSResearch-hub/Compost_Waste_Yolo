"""Registre des cartes Jetson de la flotte — administrateur uniquement.

Une carte doit être déclarée ici AVANT son premier envoi : POST
/api/sync/upload refuse toute carte inconnue ou désactivée. Pas de
suppression : désactiver (`is_active: false`) retire la carte du service en
conservant ses sessions et leur origine (sessions.jetson_id).
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from ..deps import get_db, require_roles
from ..models import JetsonDevice, normaliser_jetson_id
from ..schemas import JetsonCreate, JetsonOut, JetsonPatch

router = APIRouter(
    prefix="/api/jetsons",
    tags=["jetsons"],
    dependencies=[Depends(require_roles("administrateur"))],
)


@router.post("", response_model=JetsonOut, status_code=201)
def create_jetson(body: JetsonCreate, db=Depends(get_db)):
    device = JetsonDevice(id=body.id, name=body.name)
    db.add(device)
    try:
        db.flush()
    except IntegrityError:
        raise HTTPException(status_code=409, detail="Carte déjà déclarée")
    db.refresh(device)  # created_at/updated_at/is_active posés par la base
    return device


@router.get("", response_model=list[JetsonOut])
def list_jetsons(db=Depends(get_db)):
    return db.scalars(select(JetsonDevice).order_by(JetsonDevice.id)).all()


@router.patch("/{jetson_id}", response_model=JetsonOut)
def patch_jetson(jetson_id: str, body: JetsonPatch, db=Depends(get_db)):
    # Même normalisation qu'à la création : l'admin peut taper la MAC avec `:`
    device = db.get(JetsonDevice, normaliser_jetson_id(jetson_id))
    if device is None:
        raise HTTPException(status_code=404, detail="Carte inconnue")
    if "name" in body.model_fields_set:  # null explicite = effacer
        device.name = body.name
    if body.is_active is not None:
        device.is_active = body.is_active
    db.flush()
    db.refresh(device)  # updated_at recalculé par la base
    return device
