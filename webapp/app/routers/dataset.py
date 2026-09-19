"""Écran « Dataset » — administrateur : récupérer les données d'entraînement
et déposer des images brutes depuis son PC, sans passer par un chemin côté
serveur (contrairement à l'écran Technique).

GET /api/dataset/export — un ZIP construit À LA VOLÉE, écrit dans la réponse
au fur et à mesure (rien n'est posé sur le disque du serveur, la mémoire
tenue est celle d'une image) :

    images/       les images annotées (statut annotee|relue), le fichier
                  auquel les boîtes se rapportent (crop s'il existe)
    labels/       un .txt YOLO par image, même stem — boîtes validées seules,
                  VIDE (0 octet) pour une image sans intrus
    data.yaml     le référentiel de classes du serveur (DATA_YAML_PATH), tel
                  quel — les class_id des labels sont ses index
    groups.csv    stem,group_id (id de session) et
    classes.txt   comme l'export vers un répertoire : la sortie décompressée
                  se donne telle quelle à compost-yolo/scripts/prepare_dataset.py
    rapport.txt   ce que contient l'archive (images, boîtes, par classe, par
                  session, renommages) — le navigateur ne peut rien afficher
                  d'autre qu'un téléchargement

Même périmètre et mêmes règles que POST /api/exports : c'est `planifier_export`
(exporter.py) qui décide de ce qui sort, dans les deux cas — une image dont
le fichier est absent du stockage est ignorée et comptée
(`fichiers_manquants`, dans rapport.txt), l'export reste complet pour le
reste. Ce qui peut échouer (référentiel, class_id hors référentiel : 400) est
vérifié AVANT le premier octet envoyé : un début de téléchargement est un
téléchargement complet, sauf incident disque.

GET /api/dataset/resume — le même rapport, sans lire le contenu des fichiers
(seule leur existence est vérifiée) : pour que l'écran annonce ce que le
bouton va télécharger, images ignorées comprises.

POST /api/dataset/import — formulaire multipart, champ `archive` : un ZIP
d'images brutes (.jpg/.jpeg/.png), décompressé à plat dans un dossier
temporaire avec les gardes des envois de cartes (plafonds SYNC_MAX_UPLOAD_MB
et SYNC_MAX_UNZIPPED_MB : 413 ; archive illisible ou chemin interdit : 400),
puis importé par `import_manuel_admin` dans la session « Import_Manuel_Admin »
(créée au premier dépôt, rejointe ensuite) : les images entrent dans son lot
« import » au statut en_attente_preannotation et suivent la file de
pré-annotation. 201 + rapport d'import ; 409 + rapport si rien n'est
importable (tout en doublon) ; 409 simple si deux dépôts ont créé la session
en même temps (réessayer). Le dossier temporaire est supprimé dans tous les
cas.
"""
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Annotated, Iterator

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from ..config import get_settings
from ..deps import get_db, require_roles
from ..exporter import (PlanExport, contenu_classes_txt, contenu_groups_csv,
                        contenu_label, planifier_export)
from ..importer import SESSION_IMPORT_MANUEL, import_manuel_admin
from ..models import CaptureSession, User
from ..storage import Storage, get_storage
from .exports import SessionStatOut
from .imports import ImportReportOut, _to_out
from .sync import ArchiveInvalide, ArchiveTropGrosse, _copier_upload, _extraire_a_plat

# une seule dépendance partagée : le contrôle de rôle est mis en cache pour
# la requête, l'endpoint d'import la réutilise pour connaître l'administrateur
_admin = require_roles("administrateur")

router = APIRouter(prefix="/api/dataset", tags=["dataset"],
                   dependencies=[Depends(_admin)])

_CHUNK = 1024 * 1024


class ResumeDatasetOut(BaseModel):
    images: int
    boxes: int
    empty_labels: int
    sessions: list[SessionStatOut]
    class_counts: dict[str, int]
    renamed: list[tuple[str, str]]
    # images exportables ignorées, fichier absent du stockage
    fichiers_manquants: int


def _resume(plan: PlanExport) -> ResumeDatasetOut:
    r = plan.report
    return ResumeDatasetOut(
        images=r.images, boxes=r.boxes, empty_labels=r.empty_labels,
        sessions=[SessionStatOut(id=s[0], name=s[1], images=s[2], boxes=s[3])
                  for s in r.sessions],
        class_counts=r.class_counts, renamed=r.renamed,
        fichiers_manquants=r.fichiers_manquants,
    )


def _planifier(db, storage: Storage) -> PlanExport:
    try:
        return planifier_export(db, storage)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ── Export : ZIP à la volée ──────────────────────────────────────────────────

class _FluxSortie:
    """Destination d'écriture non positionnable pour zipfile : ce que ZipFile
    écrit s'accumule ici, le générateur de la réponse le vide après chaque
    entrée. zipfile n'a besoin que de write() et tell() ; sans seek(), il
    passe de lui-même en mode flux (descripteurs de données après chaque
    fichier) et l'archive reste valide."""

    def __init__(self) -> None:
        self._morceaux: list[bytes] = []
        self._position = 0

    def write(self, donnees) -> int:
        self._morceaux.append(bytes(donnees))
        self._position += len(donnees)
        return len(donnees)

    def tell(self) -> int:
        return self._position

    def flush(self) -> None:
        pass

    def vider(self) -> bytes:
        donnees = b"".join(self._morceaux)
        self._morceaux.clear()
        return donnees


def _entree(nom: str, compress_type: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(nom, date_time=time.localtime()[:6])
    info.compress_type = compress_type
    return info


def _flux_zip(plan: PlanExport, storage: Storage, data_yaml: bytes) -> Iterator[bytes]:
    """Génère l'archive morceau par morceau. Les JPEG sont déjà compressés :
    stockés tels quels (ZIP_STORED) ; les textes sont compressés."""
    sortie = _FluxSortie()
    with zipfile.ZipFile(sortie, "w") as zf:
        for stem, final, rel, _sid, lot in plan.entrees:
            with storage.open(rel) as src, \
                    zf.open(_entree(f"images/{final}", zipfile.ZIP_STORED), "w") as dst:
                while chunk := src.read(_CHUNK):
                    dst.write(chunk)
            zf.writestr(_entree(f"labels/{stem}.txt", zipfile.ZIP_DEFLATED),
                        contenu_label(lot).encode("utf-8"))
            yield sortie.vider()
        zf.writestr(_entree("data.yaml", zipfile.ZIP_DEFLATED), data_yaml)
        zf.writestr(_entree("groups.csv", zipfile.ZIP_DEFLATED),
                    contenu_groups_csv(plan.entrees).encode("utf-8"))
        zf.writestr(_entree("classes.txt", zipfile.ZIP_DEFLATED),
                    contenu_classes_txt(plan.noms_classes).encode("utf-8"))
        zf.writestr(_entree("rapport.txt", zipfile.ZIP_DEFLATED),
                    (plan.report.summary() + "\n").encode("utf-8"))
    # fermeture du ZipFile : répertoire central, fin d'archive
    yield sortie.vider()


@router.get("/resume", response_model=ResumeDatasetOut)
def resume(db=Depends(get_db)):
    """Ce que l'export contiendrait, sans lire le contenu du stockage."""
    if db.scalar(select(func.count()).select_from(CaptureSession)) == 0:
        # base vierge : rien à annoncer, ce n'est pas une erreur
        return ResumeDatasetOut(images=0, boxes=0, empty_labels=0, sessions=[],
                                class_counts={}, renamed=[], fichiers_manquants=0)
    return _resume(_planifier(db, get_storage()))


@router.get("/export")
def export(db=Depends(get_db)):
    storage = get_storage()
    plan = _planifier(db, storage)
    try:
        data_yaml = Path(get_settings().data_yaml_path).read_bytes()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"data.yaml illisible : {exc}")
    plan.report.output_dir = "archive ZIP (téléchargement)"
    nom = f"dataset_yolo_{datetime.now():%Y-%m-%d_%H%M}.zip"
    return StreamingResponse(
        _flux_zip(plan, storage, data_yaml),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{nom}"'},
    )


# ── Import : ZIP d'images brutes depuis le PC ────────────────────────────────

@router.post("/import", response_model=ImportReportOut, status_code=201)
def importer(
    archive: Annotated[UploadFile, File(description="ZIP d'images brutes (.jpg/.jpeg/.png)")],
    db=Depends(get_db),
    admin: User = Depends(_admin),
):
    settings = get_settings()
    with tempfile.TemporaryDirectory(prefix="compost_dataset_") as tmp:
        tmp_dir = Path(tmp)
        zip_path = tmp_dir / "depot.zip"
        images_dir = tmp_dir / "images"
        images_dir.mkdir()
        try:
            _copier_upload(archive, zip_path, settings.sync_max_upload_mb * _CHUNK)
            _extraire_a_plat(zip_path, images_dir,
                             settings.sync_max_unzipped_mb * _CHUNK)
            zip_path.unlink()  # libère la place avant les copies vers le stockage
            report = import_manuel_admin(
                db, get_storage(), source_dir=images_dir, admin_id=admin.id,
                # le poste de capture : le compte qui dépose, faute de mieux
                source_label=admin.username,
            )
        except ArchiveTropGrosse as exc:
            raise HTTPException(status_code=413, detail=str(exc))
        except ArchiveInvalide as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except IntegrityError:
            # deux dépôts ont créé « Import_Manuel_Admin » en même temps : le
            # second a perdu la course, rien n'a été écrit — réessayer rattache
            raise HTTPException(
                status_code=409,
                detail=f"Dépôt concurrent dans « {SESSION_IMPORT_MANUEL} » : réessayer")
    # TemporaryDirectory a tout supprimé, succès comme échec
    if report.aborted:
        raise HTTPException(status_code=409, detail=_to_out(report).model_dump())
    return _to_out(report)
