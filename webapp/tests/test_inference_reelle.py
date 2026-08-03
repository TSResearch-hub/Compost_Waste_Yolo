"""Inférence réelle (facultatif) : ignoré si ultralytics ou les poids sont
absents — la suite standard ne dépend jamais des poids réels.
À lancer depuis un venv où requirements-worker.txt est installé."""
import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
POIDS = REPO / "weights" / "best.pt"
IMAGES = REPO / "dataset_recolte" / "images"
LABELS = REPO / "dataset_recolte" / "labels"

manque = (importlib.util.find_spec("ultralytics") is None
          or not POIDS.is_file() or not IMAGES.is_dir() or not LABELS.is_dir())


@pytest.mark.skipif(manque, reason="ultralytics ou poids/dataset réels absents")
def test_inference_reelle_produit_des_boites_exploitables(caplog):
    import logging

    from PIL import Image as PILImage

    from app.classes import class_count, class_names
    from app.inference import UltralyticsEngine

    # une image d'entraînement portant au moins une boîte annotée : si le
    # modèle n'y voit rien à 0.05, c'est un vrai problème
    cible = next(
        IMAGES / f"{lab.stem}.jpg"
        for lab in sorted(LABELS.glob("*.txt"))
        if lab.read_text().strip() and (IMAGES / f"{lab.stem}.jpg").is_file()
    )
    nc = class_count()
    with caplog.at_level(logging.WARNING, logger="preannotation"):
        moteur = UltralyticsEngine(POIDS, conf=0.05, max_det=20,
                                   expected_names=class_names())
    # chaque classe hors de portée du modèle est journalisée NOMMÉMENT
    # (boucle vide si les poids couvrent tout le référentiel)
    for nom in moteur.classes_absentes:
        assert any(nom in r.message and "JAMAIS" in r.message
                   for r in caplog.records), f"« {nom} » absent des logs"
    detections = moteur.infer(PILImage.open(cible))

    assert detections, f"aucune boîte sur {cible.name}, image annotée du train"
    for d in detections:
        assert 0 <= d.class_id < nc
        assert 0 <= d.x_center <= 1 and 0 <= d.y_center <= 1
        assert 0 < d.box_width <= 1 and 0 < d.box_height <= 1
        assert 0 <= d.confidence <= 1
    assert re.fullmatch(r"best\.pt@[0-9a-f]{8}", moteur.model_name)
