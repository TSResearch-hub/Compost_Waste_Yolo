"""Évaluation d'un modèle : mAP Ultralytics + métriques custom du projet.

Le script déroule cinq étapes, une fonction par étape (voir main) :
1. metrics_ultralytics   mAP standard -> <run>/ultralytics_val/
2. collect_predictions   vérités terrain + prédictions brutes, image par image
3. report_per_class      rappel/précision par classe -> console + CSV
4. report_image_level    métrique d'alerte niveau image -> console + JSON,
                         et matrices de confusion par classe regroupées dans
                         UNE figure -> confusion_matrices.png
5. sweep_thresholds      (option --sweep-thresholds) calibration des seuils

Conventions (les mêmes que src/compost_detection/metrics.py) :
- gts[i]   = vérité terrain de l'image i : liste de (class_id, box) ;
- preds[i] = prédictions de l'image i : liste de (class_id, conf, box) ;
- box = (cx, cy, w, h) normalisé entre 0 et 1.

Exemple :
    python scripts/evaluate.py --weights runs/train_xxx/weights/best.pt --split test
"""

import argparse
import csv
import json
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

from compost_detection.alert import is_alert, load_alert_rules
from compost_detection.naming import create_run_dir
from compost_detection.metrics import (
    image_alert_confusion,
    mean_intruder_detections_on_negatives,
    per_class_metrics,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def fmt(value):
    """Formate une métrique qui peut être None (dénominateur nul)."""
    return f"{value:.3f}" if value is not None else "  -  "


def read_yolo_labels(label_path):
    """Lit un fichier de labels YOLO. Retourne [(class_id, (cx, cy, w, h))]."""
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        if line.strip():
            class_id, cx, cy, w, h = line.split()
            boxes.append((int(class_id), (float(cx), float(cy), float(w), float(h))))
    return boxes


def metrics_ultralytics(model, data_path, split, device, out_dir):
    """Étape 1 — métriques standard Ultralytics (mAP50, mAP50-95...).

    Tout est délégué à model.val(), qui écrit ses sorties (courbes PR, matrice
    de confusion globale...) dans <out_dir>/ultralytics_val/.
    """
    model.val(data=str(Path(data_path).resolve()), split=split, device=device,
              project=str(out_dir.resolve()), name="ultralytics_val")


def collect_predictions(model, data_cfg, split, device):
    """Étape 2 — vérités terrain et prédictions brutes pour chaque image du split.

    Prédit à confiance très basse (0.05) : les seuils sont appliqués PLUS TARD
    par les étapes 3-5, ce qui permet d'essayer plusieurs seuils sans relancer
    l'inférence. Retourne (gts, preds), listes parallèles indexées par image.
    """
    image_dir = Path(data_cfg["path"]) / data_cfg[split]
    label_dir = Path(data_cfg["path"]) / "labels" / image_dir.name
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    gts, preds = [], []
    for image_path in images:
        gts.append(read_yolo_labels(label_dir / f"{image_path.stem}.txt"))
        result = model.predict(image_path, conf=0.05, device=device, verbose=False)[0]
        preds.append([
            (int(cls), float(conf), tuple(box))
            for cls, conf, box in zip(result.boxes.cls, result.boxes.conf,
                                      result.boxes.xywhn.tolist())
        ])
    print(f"\n{len(images)} images évaluées sur le split '{split}'")
    return gts, preds


def report_per_class(gts, preds, class_names, iou, conf, out_dir):
    """Étape 3 — rappel/précision par classe, au niveau des BOÎTES.

    Chaque prédiction est appariée à une vérité terrain de même classe par
    IoU >= seuil (voir per_class_metrics). Tableau console + CSV + PNG.
    """
    stats = per_class_metrics(gts, preds, class_names, iou, conf)
    print(f"\nMétriques par classe (conf >= {conf}, IoU >= {iou}) :")
    print(f"  {'classe':<12} {'n_gt':>5} {'tp':>5} {'fp':>5} {'fn':>5} {'rappel':>7} {'précision':>9}")
    for name, s in stats.items():
        print(f"  {name:<12} {s['n_gt']:>5} {s['tp']:>5} {s['fp']:>5} {s['fn']:>5}"
              f" {fmt(s['recall']):>7} {fmt(s['precision']):>9}")
    with open(out_dir / "per_class_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "n_gt", "tp", "fp", "fn", "recall", "precision"])
        for name, s in stats.items():
            writer.writerow([name, s["n_gt"], s["tp"], s["fp"], s["fn"],
                             s["recall"], s["precision"]])
    plot_per_class_table(stats, iou, conf, out_dir / "per_class_metrics.png")
    return stats


def plot_per_class_table(stats, iou, conf, path):
    """Le tableau par classe en image (même contenu que le CSV, lisible sans tableur)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    header = ["classe", "n_gt", "tp", "fp", "fn", "rappel", "précision"]
    rows = [[name, s["n_gt"], s["tp"], s["fp"], s["fn"],
             fmt(s["recall"]), fmt(s["precision"])] for name, s in stats.items()]
    fig, ax = plt.subplots(figsize=(7, 0.45 * (len(rows) + 2)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    for j in range(len(header)):
        table[0, j].set_text_props(weight="bold")
        table[0, j].set_facecolor("#dbe5f1")
    ax.set_title(f"Métriques par classe, niveau boîte (conf >= {conf}, IoU >= {iou})",
                 fontsize=11, pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Tableau par classe (image) : {path}")


def class_image_confusion(gts, preds, class_id, conf):
    """Matrice de confusion 2x2 d'UNE classe, au niveau IMAGE.

    Croise « l'image contient au moins un objet de la classe » (vérité)
    avec « le modèle en détecte au moins un à conf >= seuil ».
    """
    gt_flags = [any(cls == class_id for cls, _ in gt) for gt in gts]
    pred_flags = [any(cls == class_id and c >= conf for cls, c, _ in pred) for pred in preds]
    return image_alert_confusion(gt_flags, pred_flags)


def plot_confusion_matrices(gts, preds, class_names, conf, path):
    """Toutes les matrices de confusion par classe dans UNE seule figure.

    Niveau image plutôt que boîte : c'est ce qui permet d'avoir un vrai TN
    (« la classe est absente et le modèle n'en voit pas »), impossible à
    définir boîte par boîte.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = 3
    rows = -(-len(class_names) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows))
    for ax, (class_id, name) in zip(axes.flat, sorted(class_names.items())):
        c = class_image_confusion(gts, preds, class_id, conf)
        matrix = [[c["tp"], c["fn"]],   # ligne « classe présente »
                  [c["fp"], c["tn"]]]   # ligne « classe absente »
        vmax = max(max(row) for row in matrix) or 1
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
        for i in range(2):
            for j in range(2):
                color = "white" if matrix[i][j] > vmax / 2 else "black"
                ax.text(j, i, matrix[i][j], ha="center", va="center", color=color)
        ax.set_xticks([0, 1], ["détectée", "non détectée"])
        ax.set_yticks([0, 1], ["présente", "absente"])
        ax.set_title(f"{name}  (rappel {fmt(c['recall'])}, précision {fmt(c['precision'])})",
                     fontsize=10)
    for ax in axes.flat[len(class_names):]:
        ax.axis("off")
    fig.suptitle(f"Confusion par classe, niveau image (conf >= {conf})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Matrices de confusion par classe : {path}")


def report_image_level(gts, preds, class_names, rules, conf, out_dir):
    """Étape 4 — la métrique qui correspond à l'usage réel : l'ALERTE.

    Une image est en alerte si une détection d'une classe intruse atteint le
    seuil de sa classe (configs/alert_rules.yaml, logique de alert.py). On la
    compare à la vérité « l'image contient un intrus ». Ajoute le taux de
    fausses détections sur les images négatives et la figure des matrices de
    confusion par classe. Retourne les résultats pour le JSON consolidé.
    """
    intruder_ids = {i for i, n in class_names.items() if n in rules}
    thresholds_by_id = {i: rules[class_names[i]] for i in intruder_ids}

    gt_flags = [any(cls in intruder_ids for cls, _ in gt) for gt in gts]
    pred_flags = [is_alert([(class_names[cls], c) for cls, c, _ in pred], rules)
                  for pred in preds]
    confusion = image_alert_confusion(gt_flags, pred_flags)
    print(f"\nMétrique niveau image (classes intruses : {', '.join(sorted(rules))}) :")
    print(f"  TP={confusion['tp']} FP={confusion['fp']} "
          f"FN={confusion['fn']} TN={confusion['tn']}")
    print(f"  rappel image    : {fmt(confusion['recall'])}")
    print(f"  précision image : {fmt(confusion['precision'])}")

    mean_fp, n_negatives = mean_intruder_detections_on_negatives(gts, preds, thresholds_by_id)
    if n_negatives:
        print(f"\nImages négatives : {n_negatives} — "
              f"{mean_fp:.2f} détection(s) intruse(s) par image en moyenne")
    else:
        print("\nAucune image négative dans ce split.")

    plot_confusion_matrices(gts, preds, class_names, conf, out_dir / "confusion_matrices.png")

    per_class_image = {name: class_image_confusion(gts, preds, class_id, conf)
                       for class_id, name in class_names.items()}
    return {
        "image_level": confusion,
        "per_class_image_level": per_class_image,
        "negatives": {"count": n_negatives, "mean_intruder_detections": mean_fp},
        "intruder_ids": intruder_ids,
    }


def sweep_thresholds(gts, preds, intruder_ids, out_dir):
    """Étape 5 (option) — calibration : rappel/précision image pour des seuils
    de 0.05 à 0.95 (le même seuil appliqué à toutes les classes intruses).
    Sert à choisir les valeurs de configs/alert_rules.yaml."""
    gt_flags = [any(cls in intruder_ids for cls, _ in gt) for gt in gts]
    with open(out_dir / "threshold_sweep.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "tp", "fp", "fn", "tn", "recall", "precision",
                         "mean_intruder_detections_on_negatives"])
        for step in range(5, 100, 5):
            t = step / 100
            uniform = {i: t for i in intruder_ids}
            flags = [any(cls in intruder_ids and conf >= t for cls, conf, _ in pred)
                     for pred in preds]
            c = image_alert_confusion(gt_flags, flags)
            mean_neg, _ = mean_intruder_detections_on_negatives(gts, preds, uniform)
            writer.writerow([t, c["tp"], c["fp"], c["fn"], c["tn"],
                             c["recall"], c["precision"], mean_neg])
    print(f"Sweep exporté : {out_dir / 'threshold_sweep.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, help="chemin du .pt à évaluer")
    parser.add_argument("--data", default="data/processed/data.yaml")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--alert-rules", default="configs/alert_rules.yaml")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="seuil de confiance des métriques custom (défaut : 0.25)")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="seuil IoU du matching prédiction/GT (défaut : 0.5)")
    parser.add_argument("--sweep-thresholds", action="store_true",
                        help="exporte rappel/précision image pour des seuils 0.05->0.95")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--device")
    args = parser.parse_args()

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    out_dir = create_run_dir(args.runs_dir, f"eval_{args.split}")
    model = YOLO(args.weights)
    data_cfg = yaml.safe_load(open(args.data, encoding="utf-8"))
    class_names = {int(k): v for k, v in data_cfg["names"].items()}
    rules = load_alert_rules(args.alert_rules)

    metrics_ultralytics(model, args.data, args.split, device, out_dir)
    gts, preds = collect_predictions(model, data_cfg, args.split, device)
    stats = report_per_class(gts, preds, class_names, args.iou, args.conf, out_dir)
    image_results = report_image_level(gts, preds, class_names, rules, args.conf, out_dir)
    if args.sweep_thresholds:
        sweep_thresholds(gts, preds, image_results["intruder_ids"], out_dir)

    # JSON consolidé : tout ce que les étapes 3 et 4 ont mesuré
    with open(out_dir / "image_level_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"split": args.split, "conf": args.conf, "iou": args.iou,
                   "per_class": stats,
                   "image_level": image_results["image_level"],
                   "per_class_image_level": image_results["per_class_image_level"],
                   "negatives": image_results["negatives"]},
                  f, indent=2, ensure_ascii=False)
    print(f"\nRésultats écrits dans {out_dir}")


if __name__ == "__main__":
    main()
