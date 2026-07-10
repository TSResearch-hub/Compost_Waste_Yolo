"""
Réentraînement du modèle YOLO sur un export du dataset annoté.

C'est la 2e moitié de la boucle du projet : l'outil d'annotation produit le
dataset, ce script produit le modèle suivant (qui pré-annotera mieux, etc.).

Étapes :
    1. Onglet « 📊 Dataset » de l'app → bouton Exporter → exports/export_.../data.yaml
    2. python train.py --data exports/export_.../data.yaml
    3. Copier le best.pt produit vers weights/best.pt (commande affichée à la fin)
       → l'app PC et le serveur mobile l'utilisent au prochain démarrage.

Par défaut le départ est weights/best.pt (fine-tuning du modèle courant) :
si le nombre de classes du data.yaml diffère (7 → 9), ultralytics adapte la
tête de détection automatiquement et conserve le reste des poids.

Sur CPU l'entraînement est long (plusieurs heures) : penser à --epochs 50
pour un premier essai, ou zipper l'export et entraîner sur Colab/GPU.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_BASE = ROOT / "weights" / "best.pt"


def main():
    parser = argparse.ArgumentParser(
        description="Réentraîne le modèle YOLO sur un export de l'onglet Dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True,
                        help="data.yaml produit par l'export (onglet Dataset)")
    parser.add_argument("--model", default=str(DEFAULT_BASE),
                        help="poids de départ (weights/best.pt = fine-tuning ; "
                             "yolo11n.pt = repartir d'un modèle pré-entraîné COCO)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640, help="taille des images d'entraînement")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=30,
                        help="arrêt anticipé après N epochs sans amélioration")
    parser.add_argument("--device", default=None, help="cuda, cpu, 0… (défaut : auto)")
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        sys.exit(f"data.yaml introuvable : {data_path}\n"
                 "→ générer un export depuis l'onglet « 📊 Dataset » de l'app.")

    # Imports tardifs : ne pas payer le chargement de torch pour un --help
    import torch
    from ultralytics import YOLO

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"composte_{datetime.now():%Y%m%d_%H%M}"

    print(f"Données   : {data_path}")
    print(f"Départ    : {args.model}")
    print(f"Device    : {device}" + ("  (⚠ CPU : compter plusieurs heures)" if device == "cpu" else ""))
    print(f"Run       : runs/{run_name}\n")

    model = YOLO(args.model)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=device,
        project=str(ROOT / "runs"),
        name=run_name,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print("\n" + "═" * 60)
    print("Entraînement terminé.")
    print(f"Meilleurs poids : {best}")
    print("Pour déployer dans l'app PC + mobile :")
    print(f"    cp {best} {DEFAULT_BASE}")
    print("(garder une copie de l'ancien best.pt si besoin de revenir en arrière)")
    print("═" * 60)


if __name__ == "__main__":
    main()
