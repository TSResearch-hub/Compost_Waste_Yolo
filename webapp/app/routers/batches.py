"""Lots : liste, assignation (verrou), libération, découpage.

Le verrou n'est PAS géré en applicatif : c'est l'index unique partiel de
batch_assignments qui garantit un seul annotateur actif par lot, y compris
sous assignations concurrentes — l'API se contente de traduire la violation
d'unicité en 409.
"""
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import and_, func, select, update
from sqlalchemy.exc import IntegrityError

from ..config import get_settings
from ..deps import get_current_user, get_db, require_roles
from ..models import (IMAGE_STATUSES, Batch, BatchAssignment, CaptureSession,
                      Image, ImageStatusEvent, User)

router = APIRouter(prefix="/api/batches", tags=["batches"])


class BatchOut(BaseModel):
    id: int
    session_id: int
    session_name: str
    name: str
    holder: str | None
    total_images: int
    done_images: int
    # Images « garées » : échec de pré-annotation au plafond de tentatives.
    # Elles ont le statut en_attente_preannotation mais ne progresseront plus
    # sans geste admin — les compter à part évite d'attendre une file morte
    parked_images: int


class AssignIn(BaseModel):
    user_id: int


class AvancementOut(BaseModel):
    batch_id: int
    total: int
    par_statut: dict[str, int]  # chaque statut présent, à zéro s'il le faut


class ReleaseIn(BaseModel):
    reason: Literal["rendu", "termine"] | None = None


class SplitIn(BaseModel):
    size: int = Field(ge=1, le=1000)


@router.get("", response_model=list[BatchOut])
def list_batches(db=Depends(get_db), user: User = Depends(get_current_user)):
    active = (
        select(BatchAssignment.batch_id, BatchAssignment.user_id)
        .where(BatchAssignment.released_at.is_(None))
        .subquery()
    )
    counts = (
        select(
            Image.batch_id,
            func.count().label("total"),
            func.count().filter(Image.status.in_(("annotee", "relue"))).label("done"),
            func.count().filter(and_(
                Image.status == "en_attente_preannotation",
                Image.preannotation_attempts
                >= get_settings().worker_max_attempts,
            )).label("parked"),
        )
        .where(Image.superseded_at.is_(None))
        .group_by(Image.batch_id)
        .subquery()
    )
    rows = db.execute(
        select(
            Batch.id, Batch.session_id, CaptureSession.name, Batch.name,
            User.username,
            func.coalesce(counts.c.total, 0), func.coalesce(counts.c.done, 0),
            func.coalesce(counts.c.parked, 0),
        )
        .join(CaptureSession, CaptureSession.id == Batch.session_id)
        .outerjoin(active, active.c.batch_id == Batch.id)
        .outerjoin(User, User.id == active.c.user_id)
        .outerjoin(counts, counts.c.batch_id == Batch.id)
        .order_by(Batch.id)
    ).all()
    return [
        BatchOut(id=r[0], session_id=r[1], session_name=r[2], name=r[3],
                 holder=r[4], total_images=r[5], done_images=r[6],
                 parked_images=r[7])
        for r in rows
    ]


@router.get("/{batch_id}/avancement", response_model=AvancementOut)
def avancement(batch_id: int, db=Depends(get_db),
               user: User = Depends(get_current_user)):
    """Compteurs par statut du lot — l'annotateur suit sa campagne, l'admin
    n'importe quel lot."""
    if db.get(Batch, batch_id) is None:
        raise HTTPException(status_code=404, detail="Lot inconnu")
    if user.role != "administrateur":
        verrou = db.scalar(
            select(BatchAssignment.id).where(
                BatchAssignment.batch_id == batch_id,
                BatchAssignment.user_id == user.id,
                BatchAssignment.released_at.is_(None),
            )
        )
        if verrou is None:
            raise HTTPException(status_code=403,
                                detail="Lot non détenu par ce compte")
    lignes = db.execute(
        select(Image.status, func.count())
        .where(Image.batch_id == batch_id, Image.superseded_at.is_(None))
        .group_by(Image.status)
    ).all()
    par_statut = {s: 0 for s in IMAGE_STATUSES}
    par_statut.update(dict(lignes))
    return AvancementOut(batch_id=batch_id, total=sum(par_statut.values()),
                         par_statut=par_statut)


@router.post("/{batch_id}/assign", status_code=201)
def assign_batch(
    batch_id: int,
    body: AssignIn,
    db=Depends(get_db),
    admin: User = Depends(require_roles("administrateur")),
):
    if db.get(Batch, batch_id) is None:
        raise HTTPException(status_code=404, detail="Lot inconnu")
    target = db.get(User, body.user_id)
    if target is None or not target.is_active:
        raise HTTPException(status_code=400, detail="Utilisateur inconnu ou inactif")

    db.add(BatchAssignment(batch_id=batch_id, user_id=target.id,
                           assigned_by=admin.id))
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        holder = db.scalar(
            select(User.username)
            .join(BatchAssignment, BatchAssignment.user_id == User.id)
            .where(BatchAssignment.batch_id == batch_id,
                   BatchAssignment.released_at.is_(None))
        )
        raise HTTPException(
            status_code=409, detail=f"Lot déjà assigné à {holder}"
        )
    return {"batch_id": batch_id, "user_id": target.id}


@router.post("/{batch_id}/release")
def release_batch(
    batch_id: int,
    body: ReleaseIn,
    db=Depends(get_db),
    user: User = Depends(get_current_user),
):
    assignment = db.scalar(
        select(BatchAssignment)
        .where(BatchAssignment.batch_id == batch_id,
               BatchAssignment.released_at.is_(None))
        .with_for_update()
    )
    if assignment is None:
        raise HTTPException(status_code=404, detail="Aucune assignation active")

    if assignment.user_id == user.id:
        reason = body.reason or "rendu"
    elif user.role == "administrateur":
        reason = "force_admin"
    else:
        raise HTTPException(status_code=403,
                            detail="Seul le détenteur ou un administrateur libère un lot")

    assignment.released_at = func.now()
    assignment.released_by = user.id
    assignment.release_reason = reason

    # Règle validée : les images restées en cours redescendent à « à annoter »,
    # événement tracé — le repreneur continue là où ça s'est arrêté
    reverted = db.scalars(
        update(Image)
        .where(Image.batch_id == batch_id, Image.status == "en_cours",
               Image.superseded_at.is_(None))
        .values(status="a_annoter")
        .returning(Image.id)
    ).all()
    for image_id in reverted:
        db.add(ImageStatusEvent(image_id=image_id, from_status="en_cours",
                                to_status="a_annoter", changed_by=user.id))
    return {"batch_id": batch_id, "reason": reason,
            "reverted_images": len(reverted)}


@router.post("/{batch_id}/split")
def split_batch(
    batch_id: int,
    body: SplitIn,
    db=Depends(get_db),
    admin: User = Depends(require_roles("administrateur")),
):
    batch = db.get(Batch, batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="Lot inconnu")
    locked = db.scalar(
        select(BatchAssignment.id)
        .where(BatchAssignment.batch_id == batch_id,
               BatchAssignment.released_at.is_(None))
    )
    if locked:
        raise HTTPException(status_code=409,
                            detail="Lot verrouillé : libérer l'assignation d'abord")

    # Seules les images non entamées bougent (et jamais les archivées).
    # Round-robin entre dossiers sources : un lot homogène par poste de
    # capture corrélerait le style d'un annotateur à un appareil
    rang = func.row_number().over(partition_by=Image.source_label,
                                  order_by=Image.id)
    eligible = db.scalars(
        select(Image.id)
        .where(Image.batch_id == batch_id, Image.superseded_at.is_(None),
               Image.status.in_(("en_attente_preannotation", "a_annoter")))
        .order_by(rang, Image.source_label)
    ).all()
    if not eligible:
        raise HTTPException(status_code=400, detail="Rien à découper dans ce lot")

    taken = set(db.scalars(
        select(Batch.name).where(Batch.session_id == batch.session_id)
    ))
    created, n = [], 1
    for start in range(0, len(eligible), body.size):
        chunk = eligible[start:start + body.size]
        while f"lot {n}" in taken:
            n += 1
        new_batch = Batch(session_id=batch.session_id, name=f"lot {n}",
                          created_by=admin.id)
        taken.add(new_batch.name)
        db.add(new_batch)
        db.flush()
        db.execute(update(Image).where(Image.id.in_(chunk))
                   .values(batch_id=new_batch.id))
        created.append({"id": new_batch.id, "name": new_batch.name,
                        "count": len(chunk)})
    return {"created": created, "moved": len(eligible)}
