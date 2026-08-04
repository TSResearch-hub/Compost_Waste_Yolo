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
import io
import os
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Response
from PIL import Image as PILImage
from PIL import ImageOps
from pydantic import BaseModel, Field
from sqlalchemy import select

from ..classes import class_count, class_names
from ..config import get_settings
from ..deps import get_current_user, get_db, require_roles
from ..models import Annotation, BatchAssignment, Image, ImageStatusEvent, User
from ..security import utcnow
from ..statuts import ORIGINES_RESTITUABLES, origines_en_cours
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
    # L'image a déjà été annotée au moins une fois (annotated_at posé) :
    # signal STABLE qui survit à la réouverture (le statut, lui, repasse
    # en_cours) — c'est lui qui distingue « négatif validé » (zéro boîte
    # mais regardée) d'une image jamais regardée
    deja_annotee: bool
    # Qui a annoté / relu, et quand — la relecture affiche l'auteur, et le
    # front en déduit si le compte courant peut relire (≠ annotateur).
    # relue_par posé ⇔ la relecture est en vigueur (une réannotation l'efface)
    annotee_par: str | None
    annotee_par_id: int | None
    annotee_le: datetime | None
    relue_par: str | None
    relue_le: datetime | None
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


class FermetureOut(BaseModel):
    statut: str


class PasserAAnnoterIn(BaseModel):
    image_ids: list[int] = Field(min_length=1)


class PasserAAnnoterOut(BaseModel):
    passees: int


@router.post("/passer-a-annoter", response_model=PasserAAnnoterOut)
def passer_a_annoter(
    body: PasserAAnnoterIn,
    db=Depends(get_db),
    admin: User = Depends(require_roles("administrateur")),
):
    """Court-circuit de la file de pré-annotation : passe des images
    `en_attente_preannotation` → `a_annoter` SANS pré-annotation (worker
    indisponible, image garée, lot qu'on choisit de ne pas pré-annoter).

    Tout-ou-rien : une image inconnue/archivée (404) ou hors file (409) fait
    échouer le lot entier, rien n'est écrit. L'événement porte l'admin comme
    auteur — `changed_by` NULL est réservé au worker : on doit pouvoir
    distinguer « le worker est passé » de « quelqu'un a court-circuité la
    file ». Les compteurs/motifs d'échec de pré-annotation sont laissés en
    place : ils documentent pourquoi le court-circuit a eu lieu."""
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
        db.add(ImageStatusEvent(
            image_id=image.id, from_status="en_attente_preannotation",
            to_status="a_annoter", changed_by=admin.id))
        image.status = "a_annoter"
    return PasserAAnnoterOut(passees=len(actives))


class ClassesOut(BaseModel):
    classes: list[str]  # référentiel data.yaml, index = class_id


@router.get("/classes", response_model=ClassesOut)
def classes(user: User = Depends(get_current_user)):
    """Le référentiel des classes (data.yaml, seule source de vérité) — pour
    les écrans qui en ont besoin sans image ouverte (filtre de la liste)."""
    return ClassesOut(classes=class_names())


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


def _nom_compte(db, user_id: int | None) -> str | None:
    if user_id is None:
        return None
    u = db.get(User, user_id)
    return (u.display_name or u.username) if u else None


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
        deja_annotee=image.annotated_at is not None,
        annotee_par=_nom_compte(db, image.annotated_by),
        annotee_par_id=image.annotated_by, annotee_le=image.annotated_at,
        relue_par=_nom_compte(db, image.reviewed_by),
        relue_le=image.reviewed_at,
        largeur=largeur, hauteur=hauteur, classes=class_names(),
        boites=[
            BoiteOut(id=a.id, class_id=a.class_id, x_center=a.x_center,
                     y_center=a.y_center, box_width=a.box_width,
                     box_height=a.box_height, source=a.source, state=a.state,
                     confidence=a.confidence)
            for a in boites
        ],
    )


@router.post("/{image_id}/fermer", response_model=FermetureOut)
def fermer_image(image_id: int, db=Depends(get_db),
                 user: User = Depends(get_current_user)):
    """Referme une image quittée SANS enregistrement : retour à son statut
    d'origine — `annotee` ou `relue` si elle l'était, `a_annoter` sinon — lu
    dans le journal des événements (dernier passage vers `en_cours`). Sans ce
    geste, une image ouverte puis abandonnée resterait `en_cours`
    indéfiniment. Même règle que la libération d'un lot (app/statuts.py).

    Événement tracé avec l'utilisateur comme auteur. Les annotations ne sont
    pas touchées (une fermeture n'est pas un enregistrement) ; le front est
    libre d'appeler sans corps ni en-tête (sendBeacon à la fermeture de la
    page)."""
    image = _acces_image(db, user, image_id)
    if image.status != "en_cours":
        raise HTTPException(
            status_code=409,
            detail=f"Image en statut « {image.status} » : rien à refermer",
        )
    origine = origines_en_cours(db, [image.id]).get(image.id)
    if origine not in ORIGINES_RESTITUABLES:
        raise HTTPException(
            status_code=409,
            detail=f"Statut d'origine « {origine} » : fermeture impossible "
                   "(transition hors liste blanche)",
        )
    db.add(ImageStatusEvent(image_id=image.id, from_status="en_cours",
                            to_status=origine, changed_by=user.id))
    image.status = origine
    return FermetureOut(statut=origine)


def _chemin_reduction(image: Image) -> Path:
    """Emplacement de la réduction en cache — répertoire dédié, jamais mêlé
    aux originaux. Clé stable : le crop est immuable dès la première
    annotation (CW001) et un recadrage crée une NOUVELLE ligne images."""
    s = get_settings()
    base = s.reduction_cache_dir or (s.storage_root / ".cache_reduites")
    genre = "crop" if image.cropped_path else "orig"
    return Path(base) / f"{image.id}_{genre}_{s.reduction_max_cote}.jpg"


@router.get("/{image_id}/fichier")
def fichier_image(image_id: int, original: bool = False, db=Depends(get_db),
                  user: User = Depends(get_current_user)):
    """Le fichier auquel les coordonnées se rapportent : crop sinon original.

    Par défaut, une VERSION RÉDUITE (côté max `REDUCTION_MAX_COTE`, JPEG,
    cache disque généré au premier accès) est servie dès que le fichier
    dépasse ce côté : les coordonnées étant normalisées [0,1], la taille
    servie n'a aucun effet sur les boîtes — le canvas se cale sur les
    dimensions réelles du bitmap reçu. `?original=true` sert l'original
    intact (réduction insuffisante, vérification). Les originaux ne sont
    jamais modifiés."""
    image = _acces_image(db, user, image_id)
    rel = image.cropped_path or image.original_path
    largeur, hauteur = (
        (image.cropped_width, image.cropped_height)
        if image.cropped_path else (image.width, image.height)
    )
    s = get_settings()
    if original or max(largeur, hauteur) <= s.reduction_max_cote:
        media = _MEDIA_TYPES.get(Path(rel).suffix.lower(),
                                 "application/octet-stream")
        return Response(content=get_storage().read(rel), media_type=media)

    cache = _chemin_reduction(image)
    if not cache.exists():
        # Génération à la demande, écriture atomique : deux requêtes
        # concurrentes produisent le même contenu, le rename est sans danger.
        # L'orientation EXIF est cuite dans les pixels pour que la réduction
        # s'affiche exactement comme l'original s'affiche dans un navigateur.
        im = PILImage.open(io.BytesIO(get_storage().read(rel)))
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((s.reduction_max_cote, s.reduction_max_cote),
                     PILImage.Resampling.LANCZOS)
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(f"{cache.name}.{os.getpid()}.part")
        im.save(tmp, "JPEG", quality=85)
        os.replace(tmp, cache)
    return Response(content=cache.read_bytes(), media_type="image/jpeg")


def _appliquer_boites(db, image: Image, boites: list[BoiteIn], user: User,
                      now) -> tuple[int, int, int]:
    """Applique l'ÉTAT COMPLET des boîtes (logique commune à l'enregistrement
    et à la relecture) : chaque proposition doit être tranchée, les boîtes du
    modèle ne sont jamais supprimées, une boîte humaine retirée l'est.
    Renvoie (validées, rejetées, supprimées). Lève 422 sans rien écrire."""
    nc = class_count()
    for b in boites:
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
    vues: set[int] = set()
    validees = rejetees = supprimees = 0
    for b in boites:
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
    return validees, rejetees, supprimees


@router.put("/{image_id}/annotations", response_model=EnregistrementOut)
def enregistrer(image_id: int, body: EnregistrementIn, db=Depends(get_db),
                user: User = Depends(get_current_user)):
    image = _acces_image(db, user, image_id)
    if image.status != "en_cours":
        raise HTTPException(
            status_code=409,
            detail=f"Image en statut « {image.status} » : l'ouvrir d'abord",
        )
    now = utcnow()
    validees, rejetees, supprimees = _appliquer_boites(
        db, image, body.boites, user, now)

    db.add(ImageStatusEvent(image_id=image.id, from_status="en_cours",
                            to_status="annotee", changed_by=user.id))
    image.status = "annotee"
    image.annotated_by, image.annotated_at = user.id, now
    # Une réannotation PÉRIME la relecture : le contenu a changé après le
    # passage du relecteur (et un relecteur qui réannote deviendrait à la
    # fois annotateur et relecteur, ce que la base interdit)
    image.reviewed_by, image.reviewed_at = None, None
    return EnregistrementOut(statut="annotee", validees=validees,
                             rejetees=rejetees, supprimees=supprimees)


@router.post("/{image_id}/relire", response_model=EnregistrementOut)
def relire(image_id: int, body: EnregistrementIn, db=Depends(get_db),
           user: User = Depends(require_roles("annotateur_confirme",
                                              "administrateur"))):
    """Relecture d'une image annotée par QUELQU'UN D'AUTRE : le relecteur
    corrige les boîtes ou valide en l'état (le corps est l'état complet,
    mêmes règles que l'enregistrement), et l'image passe `relue` avec son
    relecteur et son horodatage. L'auteur de l'annotation n'est PAS modifié :
    les retouches du relecteur sont tracées boîte par boîte (updated_by /
    decided_by). Ponctuelle et facultative : une image `annotee` est
    exportable telle quelle, rien n'oblige à la relire."""
    image = _acces_image(db, user, image_id)
    if image.status != "en_cours":
        raise HTTPException(
            status_code=409,
            detail=f"Image en statut « {image.status} » : l'ouvrir d'abord",
        )
    if image.annotated_by is None:
        raise HTTPException(
            status_code=409,
            detail="Image jamais annotée : rien à relire — l'annoter d'abord",
        )
    if image.annotated_by == user.id:
        raise HTTPException(
            status_code=409,
            detail="On ne relit pas sa propre annotation "
                   "(relecteur ≠ annotateur, garanti aussi par la base)",
        )
    now = utcnow()
    validees, rejetees, supprimees = _appliquer_boites(
        db, image, body.boites, user, now)

    db.add(ImageStatusEvent(image_id=image.id, from_status="en_cours",
                            to_status="relue", changed_by=user.id))
    image.status = "relue"
    image.reviewed_by, image.reviewed_at = user.id, now
    return EnregistrementOut(statut="relue", validees=validees,
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
