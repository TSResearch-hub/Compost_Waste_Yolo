"""Distribution des poids du modèle IA aux cartes Jetson.

Circuit :
1. un administrateur PUBLIE une version — POST /api/models_ia/upload,
   formulaire multipart : `version_name`, `pt_file` (poids `.pt`), `yaml_file`
   (le `data.yaml` de l'entraînement). Les deux fichiers sont posés dans le
   stockage sous `models/{version_name}/` (`model.pt`, `data.yaml`) et une
   ligne `model_versions` est créée ;
2. les cartes, qui n'ont pas de compte, présentent le jeton partagé
   SYNC_TOKEN (`Authorization: Bearer …`, comme pour POST /api/sync/upload)
   pour lire GET /api/models_ia/latest — la version la plus récente, avec les
   URL de ses deux fichiers — puis télécharger
   GET /api/models_ia/{id}/fichier/pt et …/fichier/yaml. Exemple côté carte :

    H="Authorization: Bearer $SYNC_TOKEN"
    curl -sS -H "$H" https://VPS/api/models_ia/latest      # → version_name, pt_url, yaml_url
    curl -sS -H "$H" -o model.pt  https://VPS/api/models_ia/3/fichier/pt
    curl -sS -H "$H" -o data.yaml https://VPS/api/models_ia/3/fichier/yaml

Une version n'est jamais modifiée ni supprimée (le stockage ne réécrit
jamais) : republier = publier sous un nouveau nom. Une carte compare le
`version_name` de `latest` au sien et ne télécharge que s'il diffère.

Ordre des vérifications à la publication, avant toute écriture : droits
(401/403), nom de version (422), extensions (422), nom déjà pris (409). Les
fichiers sont ensuite copiés dans un dossier temporaire sous plafond
MODELS_MAX_UPLOAD_MB (413), le data.yaml est contrôlé (liste `names` non
vide : 422 sinon — une carte ne doit pas recevoir un référentiel illisible),
puis copiés dans le stockage et la ligne insérée. Tout échec après la copie
efface les fichiers écrits : jamais de fichiers orphelins d'une ligne, ni de
ligne sans fichiers.
"""
import os
import tempfile
from pathlib import Path
from typing import Annotated, Literal

import yaml
from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from starlette.background import BackgroundTask

from ..classes import lire_data_yaml
from ..config import get_settings
from ..deps import get_db, require_roles
from ..models import MODEL_VERSION_NAME_REGEX, ModelVersion, User
from ..schemas import ModelVersionOut
from ..storage import get_storage
from .sync import _verifier_jeton

router = APIRouter(prefix="/api/models_ia", tags=["models_ia"])

_CHUNK = 1024 * 1024
DOSSIER_MODELES = "models"
NOM_PT = "model.pt"
NOM_YAML = "data.yaml"
# type MIME et nom de fichier proposé au téléchargement, par sorte de fichier
_FICHIERS = {
    "pt": ("application/octet-stream", NOM_PT),
    "yaml": ("application/yaml", NOM_YAML),
}


def _jeton_sync(authorization: Annotated[str | None, Header()] = None) -> None:
    """Même jeton que POST /api/sync/upload (503 non configuré, 401 invalide),
    en en-tête `Authorization: Bearer …` seulement : un GET n'a pas de corps,
    et un jeton dans l'URL finirait dans les journaux."""
    _verifier_jeton(authorization, None)


def _copier_sous_plafond(upload: UploadFile, cible: Path, max_bytes: int) -> int:
    """Copie le corps reçu vers `cible` par morceaux, sous plafond — on ne
    fait pas confiance à Content-Length. Retourne la taille écrite."""
    total = 0
    with cible.open("wb") as dst:
        while chunk := upload.file.read(_CHUNK):
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"{upload.filename} : fichier plus gros que "
                           "MODELS_MAX_UPLOAD_MB")
            dst.write(chunk)
    return total


def _chemins(version_name: str) -> tuple[str, str]:
    return (f"{DOSSIER_MODELES}/{version_name}/{NOM_PT}",
            f"{DOSSIER_MODELES}/{version_name}/{NOM_YAML}")


def _ordre_recent():
    # created_at = now() de la transaction : deux publications qui se croisent
    # peuvent partager l'instant — l'id départage, de façon déterministe
    return (ModelVersion.created_at.desc(), ModelVersion.id.desc())


@router.post("/upload", response_model=ModelVersionOut, status_code=201)
def upload(
    version_name: Annotated[str, Form(pattern=MODEL_VERSION_NAME_REGEX)],
    pt_file: Annotated[UploadFile, File(description="Poids du modèle (.pt)")],
    yaml_file: Annotated[UploadFile, File(description="data.yaml de l'entraînement")],
    db=Depends(get_db),
    admin: User = Depends(require_roles("administrateur")),
):
    if not (pt_file.filename or "").lower().endswith(".pt"):
        raise HTTPException(
            status_code=422, detail="pt_file : extension .pt attendue")
    if not (yaml_file.filename or "").lower().endswith((".yaml", ".yml")):
        raise HTTPException(
            status_code=422, detail="yaml_file : extension .yaml attendue")
    if db.scalar(select(ModelVersion.id)
                 .where(ModelVersion.version_name == version_name)) is not None:
        raise HTTPException(
            status_code=409, detail=f"Version déjà publiée : {version_name}")

    storage = get_storage()
    pt_rel, yaml_rel = _chemins(version_name)
    if storage.exists(pt_rel) or storage.exists(yaml_rel):
        # ligne absente mais fichiers présents : reste d'un incident à
        # examiner à la main — le stockage ne réécrit jamais
        raise HTTPException(
            status_code=409,
            detail=f"Le stockage contient déjà des fichiers pour {version_name}")

    max_bytes = get_settings().models_max_upload_mb * _CHUNK
    with tempfile.TemporaryDirectory(prefix="compost_modele_") as tmp:
        tmp_pt, tmp_yaml = Path(tmp) / NOM_PT, Path(tmp) / NOM_YAML
        if _copier_sous_plafond(pt_file, tmp_pt, max_bytes) == 0:
            raise HTTPException(status_code=422, detail="pt_file : fichier vide")
        _copier_sous_plafond(yaml_file, tmp_yaml, max_bytes)
        try:
            lire_data_yaml(tmp_yaml)
        except (ValueError, yaml.YAMLError) as exc:  # UnicodeDecodeError inclus
            raise HTTPException(
                status_code=422,
                detail=f"yaml_file : data.yaml invalide — {exc}")
        storage.save_file(pt_rel, tmp_pt)
        try:
            storage.save_file(yaml_rel, tmp_yaml)
        except Exception:
            storage.delete(pt_rel, missing_ok=True)
            raise
    # TemporaryDirectory a tout supprimé, succès comme échec

    version = ModelVersion(
        version_name=version_name, pt_file_path=pt_rel, yaml_file_path=yaml_rel,
        created_by=admin.id,
    )
    db.add(version)
    try:
        db.flush()
    except IntegrityError:
        # deux publications du même nom se sont croisées : la seconde a perdu
        # la course APRÈS avoir écrit ses fichiers — mais save_file ayant
        # refusé d'écraser, ce sont forcément les siens qu'on efface ici
        storage.delete(pt_rel, missing_ok=True)
        storage.delete(yaml_rel, missing_ok=True)
        raise HTTPException(
            status_code=409, detail=f"Version déjà publiée : {version_name}")
    db.refresh(version)  # created_at posé par la base
    return version


@router.get("", response_model=list[ModelVersionOut],
            dependencies=[Depends(require_roles("administrateur"))])
def list_versions(db=Depends(get_db)):
    """Toutes les versions publiées, la plus récente en premier — pour l'écran
    d'administration (cookie de session, pas de jeton)."""
    return db.scalars(select(ModelVersion).order_by(*_ordre_recent())).all()


@router.get("/latest", response_model=ModelVersionOut,
            dependencies=[Depends(_jeton_sync)])
def latest(db=Depends(get_db)):
    """La version la plus récente (created_at DESC), pour les cartes."""
    version = db.scalar(select(ModelVersion).order_by(*_ordre_recent()).limit(1))
    if version is None:
        raise HTTPException(status_code=404, detail="Aucune version de modèle publiée")
    return version


@router.get("/{version_id}/fichier/{kind}", dependencies=[Depends(_jeton_sync)])
def fichier(version_id: int, kind: Literal["pt", "yaml"], db=Depends(get_db)):
    """Le fichier lui-même, servi par morceaux (un .pt fait des dizaines de
    Mo) avec Content-Length : la carte détecte un téléchargement tronqué."""
    version = db.get(ModelVersion, version_id)
    if version is None:
        raise HTTPException(status_code=404, detail="Version inconnue")
    rel = version.pt_file_path if kind == "pt" else version.yaml_file_path
    storage = get_storage()
    if not storage.exists(rel):
        # la ligne existe, pas le fichier : incohérence côté serveur, à
        # examiner — pas un 404 que la carte prendrait pour « pas de modèle »
        raise HTTPException(
            status_code=500, detail=f"Fichier absent du stockage : {rel}")
    media, nom = _FICHIERS[kind]
    fh = storage.open(rel)
    fh.seek(0, os.SEEK_END)
    taille = fh.tell()
    fh.seek(0)
    return StreamingResponse(
        iter(lambda: fh.read(_CHUNK), b""),
        media_type=media,
        headers={"Content-Length": str(taille),
                 "Content-Disposition": f'attachment; filename="{nom}"'},
        background=BackgroundTask(fh.close),
    )
