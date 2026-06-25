"""Tests des métriques custom (cas simples construits à la main)."""

import pytest

from compost_detection.alert import is_alert
from compost_detection.metrics import (
    image_alert_confusion,
    iou,
    mean_intruder_detections_on_negatives,
    per_class_metrics,
)

CLASS_NAMES = {0: "Carton", 1: "Plastique"}
BOX = (0.5, 0.5, 0.2, 0.2)
BOX_FAR = (0.1, 0.1, 0.1, 0.1)  # IoU nul avec BOX
RULES = {"Plastique": 0.40}


def test_iou_identical_and_disjoint_boxes():
    assert iou(BOX, BOX) == pytest.approx(1.0)
    assert iou(BOX, BOX_FAR) == 0.0


def test_per_class_intruder_detected():
    gts = [[(1, BOX)]]
    preds = [[(1, 0.9, BOX)]]
    stats = per_class_metrics(gts, preds, CLASS_NAMES)
    assert stats["Plastique"] == {
        "n_gt": 1, "tp": 1, "fp": 0, "fn": 0, "recall": 1.0, "precision": 1.0}


def test_per_class_intruder_missed():
    gts = [[(1, BOX)]]
    preds = [[]]
    stats = per_class_metrics(gts, preds, CLASS_NAMES)
    assert stats["Plastique"]["recall"] == 0.0
    assert stats["Plastique"]["fn"] == 1


def test_per_class_false_positive_on_negative_image():
    gts = [[]]
    preds = [[(1, 0.8, BOX)]]
    stats = per_class_metrics(gts, preds, CLASS_NAMES)
    assert stats["Plastique"]["fp"] == 1
    assert stats["Plastique"]["precision"] == 0.0


def test_per_class_wrong_location_counts_fp_and_fn():
    gts = [[(1, BOX)]]
    preds = [[(1, 0.8, BOX_FAR)]]  # bonne classe, mauvais endroit
    stats = per_class_metrics(gts, preds, CLASS_NAMES)
    assert stats["Plastique"]["fp"] == 1
    assert stats["Plastique"]["fn"] == 1


def test_per_class_ignores_predictions_below_conf():
    gts = [[(1, BOX)]]
    preds = [[(1, 0.1, BOX)]]
    stats = per_class_metrics(gts, preds, CLASS_NAMES, conf_threshold=0.25)
    assert stats["Plastique"]["tp"] == 0
    assert stats["Plastique"]["fn"] == 1


def test_image_alert_confusion_all_cases():
    # intrus détecté (TP), intrus raté (FN), fausse alerte (FP), vrai négatif (TN)
    gt_flags = [True, True, False, False]
    pred_flags = [True, False, True, False]
    c = image_alert_confusion(gt_flags, pred_flags)
    assert (c["tp"], c["fn"], c["fp"], c["tn"]) == (1, 1, 1, 1)
    assert c["recall"] == 0.5
    assert c["precision"] == 0.5


def test_is_alert_threshold_per_class():
    assert is_alert([("Plastique", 0.5)], RULES)
    assert not is_alert([("Plastique", 0.3)], RULES)  # sous le seuil
    assert not is_alert([("Carton", 0.99)], RULES)  # classe absente des règles -> ignorée
    assert not is_alert([], RULES)


def test_mean_intruder_detections_on_negatives():
    gts = [[], [], [(1, BOX)]]  # 2 images négatives, 1 positive (exclue)
    preds = [
        [(1, 0.9, BOX), (1, 0.5, BOX_FAR)],  # 2 fausses alertes
        [(1, 0.1, BOX)],                     # sous le seuil -> 0
        [(1, 0.9, BOX)],
    ]
    mean, n_negatives = mean_intruder_detections_on_negatives(gts, preds, {1: 0.4})
    assert n_negatives == 2
    assert mean == 1.0


def test_mean_intruder_detections_no_negative_images():
    mean, n_negatives = mean_intruder_detections_on_negatives(
        [[(1, BOX)]], [[(1, 0.9, BOX)]], {1: 0.4})
    assert mean is None
    assert n_negatives == 0
