"""Trace les courbes d'entraînement (loss + métriques) depuis les results.csv.

Pour chaque run : une figure de métriques (précision, rappel, mAP@50,
mAP@50-95 par epoch) et une figure de loss (train vs val, une case par
composante — YOLO : box/cls/dfl, RT-DETR : giou/cls/l1). Avec plusieurs runs,
une figure de comparaison superpose leurs métriques sur les mêmes axes.

Accepte des dossiers de run (les results*.csv y sont recousus — cas des
reprises après crash, voir train.py --backup-dir) ou des chemins de csv.
Fonctionne tel quel sur un dossier de backup Drive (backups/<run>/).

Exemples :
    python scripts/plot_curves.py runs/finetune_08-07_14h02
    python scripts/plot_curves.py runs/pretrain_10-07_09h00 runs/pretrain_11-07_08h00 \\
        --labels YOLOv8n RT-DETR-l --output figures
"""

import argparse
from pathlib import Path

from matplotlib import pyplot as plt

from compost_detection.curves import (load_run, make_compare_figure,
                                      make_losses_figure, make_metrics_figure)


def slug(label):
    """Libellé -> nom de fichier sûr."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in label)


def save(fig, path):
    fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+",
                        help="dossiers de run (ou fichiers results.csv)")
    parser.add_argument("--labels", nargs="*", default=[],
                        help="un libellé par run, pour les titres et la légende "
                             "(défaut : nom du dossier)")
    parser.add_argument("--output", default="figures",
                        help="dossier de sortie des PNG (défaut : figures/)")
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.runs):
        parser.error("--labels doit donner exactement un libellé par run")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    runs = {}
    for i, path in enumerate(args.runs):
        p = Path(path)
        label = args.labels[i] if args.labels else (p.stem if p.is_file() else p.name)
        runs[label] = load_run(p)

    print("Figures écrites :")
    for label, rows in runs.items():
        save(make_metrics_figure(rows, label), out / f"{slug(label)}_metriques.png")
        save(make_losses_figure(rows, label), out / f"{slug(label)}_loss.png")
    if len(runs) > 1:
        save(make_compare_figure(runs), out / "comparaison_metriques.png")


if __name__ == "__main__":
    main()
