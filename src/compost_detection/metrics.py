"""Métriques custom : rappel/précision par classe et métrique d'alerte niveau image.

Conventions :
- une vérité terrain (GT) d'image = liste de (class_id, box) ;
- une prédiction d'image = liste de (class_id, confidence, box) ;
- box = (cx, cy, w, h) normalisé entre 0 et 1 (format YOLO).
"""


def iou(box_a, box_b):
    """IoU entre deux boîtes au format YOLO normalisé (cx, cy, w, h)."""
    ax1, ay1 = box_a[0] - box_a[2] / 2, box_a[1] - box_a[3] / 2
    ax2, ay2 = box_a[0] + box_a[2] / 2, box_a[1] + box_a[3] / 2
    bx1, by1 = box_b[0] - box_b[2] / 2, box_b[1] - box_b[3] / 2
    bx2, by2 = box_b[0] + box_b[2] / 2, box_b[1] + box_b[3] / 2
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = box_a[2] * box_a[3] + box_b[2] * box_b[3] - inter
    return inter / union if union > 0 else 0.0


def per_class_metrics(gts, preds, class_names, iou_threshold=0.5, conf_threshold=0.25):
    """Rappel et précision par classe sur un ensemble d'images.

    gts / preds : listes parallèles (une entrée par image, voir conventions
    du module). class_names : dict {class_id: nom}.

    Matching glouton : pour chaque classe, les prédictions (confiance
    décroissante) sont associées à la GT non encore appariée de meilleur IoU
    (>= iou_threshold). Retourne {nom: {n_gt, tp, fp, fn, recall, precision}}.
    """
    stats = {name: {"n_gt": 0, "tp": 0, "fp": 0, "fn": 0} for name in class_names.values()}
    for gt, pred in zip(gts, preds):
        for class_id, name in class_names.items():
            gt_boxes = [box for cls, box in gt if cls == class_id]
            pred_boxes = sorted(
                ((conf, box) for cls, conf, box in pred
                 if cls == class_id and conf >= conf_threshold),
                key=lambda p: p[0], reverse=True,
            )
            matched = [False] * len(gt_boxes)
            for _, pred_box in pred_boxes:
                best_index, best_iou = -1, iou_threshold
                for i, gt_box in enumerate(gt_boxes):
                    if not matched[i] and iou(pred_box, gt_box) >= best_iou:
                        best_index, best_iou = i, iou(pred_box, gt_box)
                if best_index >= 0:
                    matched[best_index] = True
                    stats[name]["tp"] += 1
                else:
                    stats[name]["fp"] += 1
            stats[name]["n_gt"] += len(gt_boxes)
            stats[name]["fn"] += matched.count(False)
    for s in stats.values():
        s["recall"] = s["tp"] / s["n_gt"] if s["n_gt"] else None
        denom = s["tp"] + s["fp"]
        s["precision"] = s["tp"] / denom if denom else None
    return stats


def image_alert_confusion(gt_flags, pred_flags):
    """Matrice de confusion binaire au niveau image.

    gt_flags : par image, True si la vérité terrain contient au moins un
    intrus. pred_flags : par image, True si le modèle lève une alerte.
    Retourne {tp, fp, fn, tn, recall, precision}.
    """
    tp = sum(1 for g, p in zip(gt_flags, pred_flags) if g and p)
    fp = sum(1 for g, p in zip(gt_flags, pred_flags) if not g and p)
    fn = sum(1 for g, p in zip(gt_flags, pred_flags) if g and not p)
    tn = sum(1 for g, p in zip(gt_flags, pred_flags) if not g and not p)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": tp / (tp + fn) if (tp + fn) else None,
        "precision": tp / (tp + fp) if (tp + fp) else None,
    }


def mean_intruder_detections_on_negatives(gts, preds, thresholds_by_id):
    """Nb moyen de détections intruses par image NÉGATIVE (aucun objet annoté).

    thresholds_by_id : {class_id: seuil de confiance} pour les classes
    intruses. Retourne (moyenne ou None s'il n'y a pas d'image négative,
    nombre d'images négatives).
    """
    counts = [
        sum(1 for cls, conf, _ in pred
            if cls in thresholds_by_id and conf >= thresholds_by_id[cls])
        for gt, pred in zip(gts, preds) if not gt
    ]
    if not counts:
        return None, 0
    return sum(counts) / len(counts), len(counts)
