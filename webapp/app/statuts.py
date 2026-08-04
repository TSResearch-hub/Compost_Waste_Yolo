"""Retour d'une image `en_cours` à son statut d'origine.

Règle commune à la fermeture d'une image (POST /images/{id}/fermer) et à la
libération d'un lot (POST /batches/{id}/release) : une image quittée sans
enregistrement revient là d'où elle venait — `annotee` reste `annotee`,
`relue` reste `relue`, tout le reste redevient `a_annoter`. La rétrograder
en bloc la sortirait de l'export YOLO (statuts annotee/relue seulement),
donc de l'entraînement, sans aucun signal.

L'origine est lue dans le journal `image_status_events` (source de vérité) :
le `from_status` du dernier passage vers `en_cours`.
"""
from sqlalchemy import select

from .models import ImageStatusEvent

# Statuts qu'une fermeture peut restituer — les trois `from_status` possibles
# d'un passage vers en_cours d'après la liste blanche CW002
ORIGINES_RESTITUABLES = ("a_annoter", "annotee", "relue")


def origines_en_cours(db, image_ids) -> dict[int, str]:
    """Pour chaque image, le `from_status` de son dernier passage vers
    `en_cours`. Une image absente du résultat n'a aucun événement de ce type
    (ligne posée hors API) : son origine est inconnue."""
    ids = list(image_ids)
    if not ids:
        return {}
    lignes = db.execute(
        select(ImageStatusEvent.image_id, ImageStatusEvent.from_status)
        .where(ImageStatusEvent.image_id.in_(ids),
               ImageStatusEvent.to_status == "en_cours")
        .distinct(ImageStatusEvent.image_id)
        .order_by(ImageStatusEvent.image_id, ImageStatusEvent.id.desc())
    ).all()
    return {image_id: origine for image_id, origine in lignes
            if origine is not None}
