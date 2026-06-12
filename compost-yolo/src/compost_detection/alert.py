"""Logique de décision d'alerte au niveau image.

Les règles (classes intruses + seuil de confiance par classe) sont définies
dans configs/alert_rules.yaml.
"""

import yaml


def load_alert_rules(path):
    """Charge les règles d'alerte. Retourne {nom_de_classe: seuil}."""
    with open(path, encoding="utf-8") as f:
        rules = yaml.safe_load(f)
    return rules["intruder_thresholds"]


def is_alert(detections, intruder_thresholds):
    """Décide si une image déclenche une alerte.

    detections : liste de (nom_de_classe, confiance). Alerte si au moins une
    détection d'une classe intruse atteint le seuil de sa classe.
    """
    return any(
        class_name in intruder_thresholds and conf >= intruder_thresholds[class_name]
        for class_name, conf in detections
    )
