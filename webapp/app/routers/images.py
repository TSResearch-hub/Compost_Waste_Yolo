"""Canvas d'annotation — segment API : ouvrir, servir le fichier,
enregistrer, naviguer.

Règles (fixées le 2026-08-03) :
- un annotateur n'accède qu'aux images d'un lot dont il détient le verrou
  ACTIF ; un administrateur accède à tout ;
- l'ouverture passe l'image en `en_cours` (événement tracé). Une image encore
  en file de pré-annotation n'est pas ouvrable : CW002 n'a pas de transition
  directe en_attente_preannotation → en_cours ;
- l'enregistrement reçoit l'ÉTAT COMPLET des boîtes et passe l'image en
  `annotee`. Chaque proposition du modèle doit être tranchée — une `proposee`
  absente du corps est une erreur (422), jamais une décision implicite. Les
  rejetées sont CONSERVÉES avec la géométrie du modèle : ce sont ses faux
  positifs. Une boîte du modèle déjà validée puis retirée du canvas redevient
  rejetée — les boîtes du modèle ne sont JAMAIS supprimées ; une boîte
  humaine retirée est supprimée. Zéro boîte validée = négatif, enregistrement
  valide ;
- coordonnées YOLO normalisées, relatives au FICHIER ANNOTÉ (le crop s'il
  existe, sinon l'original) ; class_id vérifié contre data.yaml côté serveur.
"""
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field
from sqlalchemy import select

from ..classes import class_count, class_names
from ..deps import get_current_user, get_db
from ..models import Annotation, BatchAssignment, Image, ImageStatusEvent, User
from ..security import utcnow
from ..storage import get_storage

router = APIRouter(prefix="/api/images", tags=["images"])

_MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}


class BoiteOut(BaseModel):
    id: int
    class_id: int
    x_center: float
    y_center: float
    box_width: float
    box_height: float
    source: str
    state: str
    confidence: float | None


class OuvertureOut(BaseModel):
    id: int
    batch_id: int
    session_id: int
    nom: str  # export_filename — le nom affiché en évidence
    statut: str
    # dimensions du FICHIER ANNOTÉ (crop si présent) : les coordonnées
    # normalisées s'y rapportent
    largeur: int
    hauteur: int
    classes: list[str]  # référentiel data.yaml, index = class_id
    boites: list[BoiteOut]


class BoiteIn(BaseModel):
    id: int | None = None  # absent = nouvelle boîte humaine
    class_id: int
    x_center: float = Field(ge=0, le=1)
    y_center: float = Field(ge=0, le=1)
    box_width: float = Field(gt=0, le=1)
    box_height: float = Field(gt=0, le=1)
    etat: Literal["validee", "rejetee"]


class EnregistrementIn(BaseModel):
    boites: list[BoiteIn]


class EnregistrementOut(BaseModel):
    statut: str
    validees: int
    rejetees: int
    supprimees: int


class VoisineOut(BaseModel):
    id: int
    nom: str
    statut: str


class VoisinesOut(BaseModel):
    precedente: VoisineOut | None
    suivante: VoisineOut | None


def _acces_image(db, user: User, image_id: int) -> Image:
    """L'image existe, est active, et le compte y a droit : administrateur,
    ou détenteur du verrou actif sur son lot."""
    image = db.get(Image, image_id)
    if image is None or image.superseded_at is not None:
        raise HTTPException(status_code=404, detail="Image inconnue")
    if user.role != "administrateur":
        verrou = db.scalar(
            select(BatchAssignment.id).where(
                BatchAssignment.batch_id == image.batch_id,
                BatchAssignment.user_id == user.id,
                BatchAssignment.released_at.is_(None),
            )
        )
        if verrou is None:
            raise HTTPException(status_code=403,
                                detail="Lot non détenu par ce compte")
    return image


@router.post("/{image_id}/ouvrir", response_model=OuvertureOut)
def ouvrir_image(image_id: int, db=Depends(get_db),
                 user: User = Depends(get_current_user)):
    image = _acces_image(db, user, image_id)
    if image.status == "en_attente_preannotation":
        raise HTTPException(
            status_code=409,
            detail="Image encore en file de pré-annotation : pas ouvrable",
        )
    if image.status != "en_cours":  # réouverture en_cours = pas d'événement
        db.add(ImageStatusEvent(image_id=image.id, from_status=image.status,
                                to_status="en_cours", changed_by=user.id))
        image.status = "en_cours"
    largeur, hauteur = (
        (image.cropped_width, image.cropped_height)
        if image.cropped_path else (image.width, image.height)
    )
    boites = db.scalars(
        select(Annotation).where(Annotation.image_id == image.id)
        .order_by(Annotation.id)
    ).all()
    return OuvertureOut(
        id=image.id, batch_id=image.batch_id, session_id=image.session_id,
        nom=image.export_filename, statut=image.status,
        largeur=largeur, hauteur=hauteur, classes=class_names(),
        boites=[
            BoiteOut(id=a.id, class_id=a.class_id, x_center=a.x_center,
                     y_center=a.y_center, box_width=a.box_width,
                     box_height=a.box_height, source=a.source, state=a.state,
                     confidence=a.confidence)
            for a in boites
        ],
    )


@router.get("/{image_id}/fichier")
def fichier_image(image_id: int, db=Depends(get_db),
                  user: User = Depends(get_current_user)):
    """Le fichier auquel les coordonnées se rapportent : crop sinon original."""
    image = _acces_image(db, user, image_id)
    rel = image.cropped_path or image.original_path
    media = _MEDIA_TYPES.get(Path(rel).suffix.lower(),
                             "application/octet-stream")
    return Response(content=get_storage().read(rel), media_type=media)


@router.put("/{image_id}/annotations", response_model=EnregistrementOut)
def enregistrer(image_id: int, body: EnregistrementIn, db=Depends(get_db),
                user: User = Depends(get_current_user)):
    image = _acces_image(db, user, image_id)
    if image.status != "en_cours":
        raise HTTPException(
            status_code=409,
            detail=f"Image en statut « {image.status} » : l'ouvrir d'abord",
        )
    nc = class_count()
    for b in body.boites:
        if not 0 <= b.class_id < nc:
            raise HTTPException(
                status_code=422,
                detail=f"class_id {b.class_id} hors référentiel "
                       f"(data.yaml : {nc} classes)",
            )

    existantes = {
        a.id: a for a in db.scalars(
            select(Annotation).where(Annotation.image_id == image.id))
    }
    now = utcnow()
    vues: set[int] = set()
    validees = rejetees = supprimees = 0
    for b in body.boites:
        if b.id is None:
            if b.etat == "rejetee":
                raise HTTPException(
                    status_code=422,
                    detail="une boîte nouvelle ne peut pas naître rejetée")
            db.add(Annotation(
                image_id=image.id, class_id=b.class_id,
                x_center=b.x_center, y_center=b.y_center,
                box_width=b.box_width, box_height=b.box_height,
                source="humain", state="validee",
                created_by=user.id, decided_by=user.id, decided_at=now,
            ))
            validees += 1
            continue
        a = existantes.get(b.id)
        if a is None:
            raise HTTPException(status_code=422,
                                detail=f"boîte {b.id} inconnue sur cette image")
        vues.add(b.id)
        if b.etat == "rejetee":
            if a.source == "humain":
                raise HTTPException(
                    status_code=422,
                    detail="une boîte humaine ne se rejette pas : la retirer "
                           "du corps pour la supprimer",
                )
            rejetees += 1  # géométrie et classe du modèle conservées telles
        else:
            envoye = (b.class_id, b.x_center, b.y_center,
                      b.box_width, b.box_height)
            if (a.class_id, a.x_center, a.y_center,
                    a.box_width, a.box_height) != envoye:
                (a.class_id, a.x_center, a.y_center,
                 a.box_width, a.box_height) = envoye
                a.updated_by, a.updated_at = user.id, now
            validees += 1
        a.state = b.etat
        a.decided_by, a.decided_at = user.id, now
    for aid, a in existantes.items():
        if aid in vues:
            continue
        if a.state == "proposee":
            raise HTTPException(
                status_code=422,
                detail=f"boîte proposée {aid} sans décision : chaque "
                       "proposition doit être validée ou rejetée",
            )
        if a.source == "humain":
            db.delete(a)
            supprimees += 1
        elif a.state == "validee":  # boîte du modèle retirée du canvas
            a.state = "rejetee"
            a.decided_by, a.decided_at = user.id, now
            rejetees += 1
        # déjà rejetée et absente du corps : reste rejetée, telle quelle

    db.add(ImageStatusEvent(image_id=image.id, from_status="en_cours",
                            to_status="annotee", changed_by=user.id))
    image.status = "annotee"
    image.annotated_by, image.annotated_at = user.id, now
    return EnregistrementOut(statut="annotee", validees=validees,
                             rejetees=rejetees, supprimees=supprimees)


@router.get("/{image_id}/voisines", response_model=VoisinesOut)
def voisines(image_id: int, db=Depends(get_db),
             user: User = Depends(get_current_user)):
    """Image précédente et suivante dans le lot (actives, ordre par id)."""
    image = _acces_image(db, user, image_id)

    def _bord(avant: bool) -> VoisineOut | None:
        cond = Image.id < image.id if avant else Image.id > image.id
        ordre = Image.id.desc() if avant else Image.id.asc()
        ligne = db.execute(
            select(Image.id, Image.export_filename, Image.status)
            .where(Image.batch_id == image.batch_id,
                   Image.superseded_at.is_(None), cond)
            .order_by(ordre).limit(1)
        ).first()
        if ligne is None:
            return None
        return VoisineOut(id=ligne[0], nom=ligne[1], statut=ligne[2])

    return VoisinesOut(precedente=_bord(avant=True),
                       suivante=_bord(avant=False))
