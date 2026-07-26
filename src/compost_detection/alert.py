"""Logique de décision d'alerte au niveau image.

Les règles (classes intruses + seuil de confiance par classe, consignes de
retrait par classe) sont définies dans configs/alert_rules.yaml.
"""

import yaml


def load_alert_rules(path):
    """Charge les règles d'alerte. Retourne {nom_de_classe: seuil}."""
    with open(path, encoding="utf-8") as f:
        rules = yaml.safe_load(f)
    return rules["intruder_thresholds"]


def load_alert_config(path):
    """Charge la config d'alerte complète : (seuils par classe, consignes par classe).

    Les consignes (section instructions:, optionnelle) sont les instructions de
    retrait affichées à l'opérateur quand une classe déclenche une alerte.
    """
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["intruder_thresholds"], cfg.get("instructions", {})


def alerting_classes(detections, intruder_thresholds):
    """Classes intruses dont au moins une détection atteint le seuil de sa classe.

    detections : liste de (nom_de_classe, confiance). Retourne une liste triée
    (vide si rien ne déclenche).
    """
    return sorted({
        class_name for class_name, conf in detections
        if class_name in intruder_thresholds and conf >= intruder_thresholds[class_name]
    })


def is_alert(detections, intruder_thresholds):
    """Décide si une image déclenche une alerte.

    detections : liste de (nom_de_classe, confiance). Alerte si au moins une
    détection d'une classe intruse atteint le seuil de sa classe.
    """
    return bool(alerting_classes(detections, intruder_thresholds))
