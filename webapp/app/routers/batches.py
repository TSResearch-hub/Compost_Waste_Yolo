"""Lots : liste, assignation (verrou), libération, découpage.

Le verrou n'est PAS géré en applicatif : c'est l'index unique partiel de
batch_assignments qui garantit un seul annotateur actif par lot, y compris
sous assignations concurrentes — l'API se contente de traduire la violation
d'unicité en 409.
"""
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import and_, func, select, update
from sqlalchemy.exc import IntegrityError

from ..config import get_settings
from ..deps import get_current_user, get_db, require_roles
from ..models import (IMAGE_STATUSES, Annotation, Batch, BatchAssignment,
                      CaptureSession, Image, ImageStatusEvent, User)
from ..statuts import ORIGINES_RESTITUABLES, origines_en_cours

router = APIRouter(prefix="/api/batches", tags=["batches"])


class BatchOut(BaseModel):
    id: int
    session_id: int
    session_name: str
    name: str
    holder: str | None
    # depuis quand le détenteur actif tient le verrou (None sans détenteur)
    holder_since: datetime | None
    total_images: int
    done_images: int
    # Images `en_cours`, comptées À PART : ouvrir une image annotée pour la
    # relire la repasse en_cours — l'avancement ne doit pas paraître reculer
    # quand on a simplement ouvert une image (problème de présentation,
    # le modèle de données ne bouge pas)
    in_progress_images: int
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


class ImageLotOut(BaseModel):
    id: int
    nom: str  # export_filename
    statut: str
    propositions: int  # boîtes du modèle encore 'proposee' (à trancher)
    # class_id présents sur l'image (boîtes proposées ou validées — les
    # rejetées sont des faux positifs, elles ne « portent » pas la classe) :
    # sert au filtre « toutes les images portant du Composite »
    classes: list[int]


def _acces_lot(db, user: User, batch_id: int) -> None:
    """Le lot existe et le compte y a droit : administrateur, ou détenteur
    du verrou actif."""
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


@router.get("", response_model=list[BatchOut])
def list_batches(db=Depends(get_db), user: User = Depends(get_current_user)):
    active = (
        select(BatchAssignment.batch_id, BatchAssignment.user_id,
               BatchAssignment.assigned_at)
        .where(BatchAssignment.released_at.is_(None))
        .subquery()
    )
    counts = (
        select(
            Image.batch_id,
            func.count().label("total"),
            func.count().filter(Image.status.in_(("annotee", "relue"))).label("done"),
            func.count().filter(Image.status == "en_cours").label("in_progress"),
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
            User.username, active.c.assigned_at,
            func.coalesce(counts.c.total, 0), func.coalesce(counts.c.done, 0),
            func.coalesce(counts.c.in_progress, 0),
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
                 holder=r[4], holder_since=r[5], total_images=r[6],
                 done_images=r[7], in_progress_images=r[8], parked_images=r[9])
        for r in rows
    ]


@router.get("/{batch_id}/avancement", response_model=AvancementOut)
def avancement(batch_id: int, db=Depends(get_db),
               user: User = Depends(get_current_user)):
    """Compteurs par statut du lot — l'annotateur suit sa campagne, l'admin
    n'importe quel lot."""
    _acces_lot(db, user, batch_id)
    lignes = db.execute(
        select(Image.status, func.count())
        .where(Image.batch_id == batch_id, Image.superseded_at.is_(None))
        .group_by(Image.status)
    ).all()
    par_statut = {s: 0 for s in IMAGE_STATUSES}
    par_statut.update(dict(lignes))
    return AvancementOut(batch_id=batch_id, total=sum(par_statut.values()),
                         par_statut=par_statut)


@router.get("/{batch_id}/images", response_model=list[ImageLotOut])
def images_du_lot(batch_id: int, db=Depends(get_db),
                  user: User = Depends(get_current_user)):
    """Liste ordonnée des images actives du lot — la trame de la campagne :
    position dans le lot, reprise (première `en_cours` sinon `a_annoter`),
    propositions restant à trancher par image. Mêmes droits que
    l'avancement : détenteur du verrou actif, ou administrateur."""
    _acces_lot(db, user, batch_id)
    propositions = (
        select(Annotation.image_id, func.count().label("n"))
        .where(Annotation.state == "proposee")
        .group_by(Annotation.image_id)
        .subquery()
    )
    presentes = (
        select(Annotation.image_id,
               func.array_agg(Annotation.class_id.distinct()).label("cls"))
        .where(Annotation.state != "rejetee")
        .group_by(Annotation.image_id)
        .subquery()
    )
    lignes = db.execute(
        select(Image.id, Image.export_filename, Image.status,
               func.coalesce(propositions.c.n, 0), presentes.c.cls)
        .outerjoin(propositions, propositions.c.image_id == Image.id)
        .outerjoin(presentes, presentes.c.image_id == Image.id)
        .where(Image.batch_id == batch_id, Image.superseded_at.is_(None))
        .order_by(Image.id)
    ).all()
    return [ImageLotOut(id=l[0], nom=l[1], statut=l[2], propositions=l[3],
                        classes=sorted(l[4] or ()))
            for l in lignes]


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

    # Les images restées « en cours » reviennent à leur statut d'ORIGINE, lu
    # dans le journal — même règle que POST /images/{id}/fermer : une image
    # annotée rouverte puis abandonnée reste annotée, la rétrograder la
    # sortirait de l'export (donc de l'entraînement) sans aucun signal.
    # Origine inconnue (ligne posée hors API) : a_annoter, le défaut sûr.
    en_cours = db.scalars(
        select(Image).where(Image.batch_id == batch_id,
                            Image.status == "en_cours",
                            Image.superseded_at.is_(None))
    ).all()
    origines = origines_en_cours(db, (i.id for i in en_cours))
    for image in en_cours:
        origine = origines.get(image.id)
        if origine not in ORIGINES_RESTITUABLES:
            origine = "a_annoter"
        db.add(ImageStatusEvent(image_id=image.id, from_status="en_cours",
                                to_status=origine, changed_by=user.id))
        image.status = origine
    return {"batch_id": batch_id, "reason": reason,
            "reverted_images": len(en_cours)}


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
