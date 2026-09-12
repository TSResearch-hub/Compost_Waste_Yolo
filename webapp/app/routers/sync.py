"""Réception des envois automatiques des cartes Jetson.

POST /api/sync/upload — pas de cookie de session : la carte présente le jeton
partagé SYNC_TOKEN (en-tête `Authorization: Bearer …` ou champ de formulaire
`token`) et son identifiant (en-tête `X-Jetson-Id` ou champ `jetson_id`), plus
un ZIP d'images (champ `archive`). Exemple côté carte :

    curl -sS -X POST https://VPS/api/sync/upload \\
         -H "Authorization: Bearer $SYNC_TOKEN" -H "X-Jetson-Id: $(cat /sys/class/net/eth0/address)" \\
         -F captured_on=2026-09-12 -F archive=@captures.zip

Ordre des vérifications, volontairement AVANT toute lecture du fichier :
jeton (503 si non configuré, 401 sinon), identifiant (422 invalide, 403
carte inconnue ou désactivée). Puis le ZIP est copié dans un dossier
temporaire sous plafond de taille (413), décompressé À PLAT dans un
sous-dossier (413 si le contenu dépasse le plafond, 400 si l'archive est
illisible ou tente de sortir du dossier), et `import_jetson_upload` fait le
reste : session `{jetson_id}_{date}` créée ou rejointe, images en
`en_attente_preannotation`, doublons ignorés, rapport. Le dossier temporaire
est supprimé dans tous les cas, succès ou échec.

Réponses : 201 + rapport d'import ; 409 + rapport si rien n'est importable
(archive vide, tout en doublon — ré-envoi après coupure réseau) ; 409 simple
si deux envois de la même carte se sont croisés (réessayer) ; 400 sur
paramètres refusés par l'import.
"""
import re
import secrets
import tempfile
import zipfile
import zlib
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from sqlalchemy.exc import IntegrityError

from ..config import get_settings
from ..deps import get_db
from ..importer import import_jetson_upload
from ..models import JETSON_ID_REGEX, JetsonDevice, normaliser_jetson_id
from ..storage import get_storage
from .imports import ImportReportOut, _to_out

router = APIRouter(prefix="/api/sync", tags=["sync"])

_CHUNK = 1024 * 1024


class ArchiveInvalide(ValueError):
    """ZIP illisible, ou entrée qui tente de sortir du dossier d'extraction."""


class ArchiveTropGrosse(ValueError):
    """Plafond SYNC_MAX_UPLOAD_MB ou SYNC_MAX_UNZIPPED_MB dépassé."""


def _verifier_jeton(authorization: str | None, token_form: str | None) -> None:
    attendu = get_settings().sync_token
    if not attendu:
        raise HTTPException(
            status_code=503,
            detail="Réception désactivée : SYNC_TOKEN non configuré sur le serveur",
        )
    presente = token_form
    if authorization:
        scheme, _, valeur = authorization.partition(" ")
        if scheme.lower() == "bearer" and valeur.strip():
            presente = valeur.strip()
    # Comparaison en temps constant : le jeton est le seul secret de la carte
    if not presente or not secrets.compare_digest(presente, attendu):
        raise HTTPException(status_code=401, detail="Jeton invalide")


def _resoudre_carte(db, header_id: str | None, form_id: str | None) -> JetsonDevice:
    brut = header_id or form_id
    if not brut or not brut.strip():
        raise HTTPException(
            status_code=422,
            detail="Identifiant de carte requis (en-tête X-Jetson-Id ou champ jetson_id)",
        )
    jetson_id = normaliser_jetson_id(brut)
    if not re.fullmatch(JETSON_ID_REGEX, jetson_id):
        raise HTTPException(status_code=422, detail="Identifiant de carte invalide")
    device = db.get(JetsonDevice, jetson_id)
    if device is None:
        raise HTTPException(status_code=403, detail="Carte inconnue : à déclarer par un administrateur")
    if not device.is_active:
        raise HTTPException(status_code=403, detail="Carte désactivée")
    return device


def _copier_upload(upload: UploadFile, cible: Path, max_bytes: int) -> None:
    """Copie le corps reçu vers `cible` par morceaux, sous plafond — on ne
    fait pas confiance à Content-Length."""
    total = 0
    with cible.open("wb") as dst:
        while chunk := upload.file.read(_CHUNK):
            total += len(chunk)
            if total > max_bytes:
                raise ArchiveTropGrosse("archive plus grosse que SYNC_MAX_UPLOAD_MB")
            dst.write(chunk)


def _nom_a_plat(parts: tuple[str, ...], pris: set[str]) -> str:
    """Nom de fichier à plat : intact pour une entrée à la racine du ZIP,
    préfixé de ses dossiers (`a__b__nom.jpg`) pour une entrée imbriquée,
    suffixé en dernier recours — deux fichiers du même envoi ne doivent pas
    s'écraser avant même l'analyse de doublons."""
    nom = parts[-1] if len(parts) == 1 else "__".join(parts)
    candidat, k = nom, 2
    stem, ext = Path(nom).stem, Path(nom).suffix
    while candidat in pris:
        candidat = f"{stem}_{k}{ext}"
        k += 1
    pris.add(candidat)
    return candidat


def _extraire_a_plat(zip_path: Path, dest: Path, max_bytes: int) -> int:
    """Extrait les FICHIERS du ZIP à plat dans `dest` (qui existe déjà).
    Refuse chemins absolus et `..` (zip-slip) ; ignore dossiers, entrées
    cachées (`.x`) et métadonnées macOS (`__MACOSX`). Compte les octets
    réellement décompressés, pas la taille annoncée par l'en-tête. Retourne le
    nombre de fichiers extraits."""
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile:
        raise ArchiveInvalide("archive ZIP illisible")
    total, extraits = 0, 0
    pris: set[str] = set()
    try:
        with zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                chemin = PurePosixPath(info.filename.replace("\\", "/"))
                parts = chemin.parts
                if chemin.is_absolute() or not parts or any(
                    p in ("", ".", "..") for p in parts
                ):
                    raise ArchiveInvalide(
                        f"chemin interdit dans l'archive : {info.filename}")
                if any(p.startswith(".") or p == "__MACOSX" for p in parts):
                    continue
                nom = _nom_a_plat(parts, pris)
                with zf.open(info) as src, (dest / nom).open("wb") as dst:
                    while chunk := src.read(_CHUNK):
                        total += len(chunk)
                        if total > max_bytes:
                            raise ArchiveTropGrosse(
                                "contenu décompressé plus gros que "
                                "SYNC_MAX_UNZIPPED_MB")
                        dst.write(chunk)
                extraits += 1
    except (zipfile.BadZipFile, zlib.error, EOFError, RuntimeError) as exc:
        # entrée corrompue, méthode de compression inconnue, archive chiffrée
        raise ArchiveInvalide(f"archive ZIP illisible : {exc}") from exc
    return extraits


@router.post("/upload", response_model=ImportReportOut, status_code=201)
def upload(
    archive: Annotated[UploadFile, File(description="ZIP d'images (.jpg/.jpeg/.png)")],
    db=Depends(get_db),
    authorization: Annotated[str | None, Header()] = None,
    x_jetson_id: Annotated[str | None, Header(alias="X-Jetson-Id")] = None,
    token: Annotated[str | None, Form()] = None,
    jetson_id: Annotated[str | None, Form()] = None,
    # Optionnels : sans eux, session `{jetson_id}_{date du jour serveur}`.
    # `captured_on` vaut la peine d'être envoyé par la carte : autour de
    # minuit, la date du serveur peut ne pas être celle de la capture.
    session_name: Annotated[str | None, Form()] = None,
    captured_on: Annotated[date | None, Form()] = None,
    notes: Annotated[str | None, Form()] = None,
):
    _verifier_jeton(authorization, token)
    device = _resoudre_carte(db, x_jetson_id, jetson_id)
    settings = get_settings()

    with tempfile.TemporaryDirectory(prefix="compost_sync_") as tmp:
        tmp_dir = Path(tmp)
        zip_path = tmp_dir / "envoi.zip"
        # Le dossier porte l'identifiant de la carte : lisible dans les logs,
        # mais c'est `source_label=` qui fait foi côté import
        images_dir = tmp_dir / device.id
        images_dir.mkdir()
        try:
            _copier_upload(archive, zip_path, settings.sync_max_upload_mb * _CHUNK)
            _extraire_a_plat(zip_path, images_dir,
                             settings.sync_max_unzipped_mb * _CHUNK)
            zip_path.unlink()  # libère la place avant les copies vers le stockage
            report = import_jetson_upload(
                db, get_storage(), source_dir=images_dir, jetson_id=device.id,
                session_name=session_name or None, captured_on=captured_on,
                notes=notes or None,
            )
        except ArchiveTropGrosse as exc:
            raise HTTPException(status_code=413, detail=str(exc))
        except ArchiveInvalide as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except IntegrityError:
            # Deux envois de la même carte ont créé la session du jour en même
            # temps : le second a perdu la course — rien n'a été écrit, il
            # suffit de réessayer (rattachement)
            raise HTTPException(
                status_code=409,
                detail="Envoi concurrent de la même carte : réessayer")
    # TemporaryDirectory a tout supprimé, succès comme échec
    if report.aborted:
        raise HTTPException(status_code=409, detail=_to_out(report).model_dump())
    return _to_out(report)
