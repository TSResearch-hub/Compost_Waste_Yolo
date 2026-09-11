"""Fine-tuning d'un modèle YOLO ou RT-DETR pré-entraîné (Ultralytics) sur le dataset préparé.

Les hyperparamètres viennent d'un YAML de configs/ (train_default.yaml par
défaut ; finetune_rtdetr.yaml / finetune_yolo.yaml pour le fine-tuning). TOUT
le YAML, sauf `model`, est transmis tel quel à Ultralytics : une clé inconnue
fait échouer le run tout de suite, jamais silencieusement. Quelques clés sont
surchargeables en CLI. Le run est horodaté, la config utilisée est copiée
dans son dossier.

Exemples :
    python scripts/train.py
    python scripts/train.py --config configs/finetune_rtdetr.yaml --model models/v0_pretrain_rtdetr-l.pt
    python scripts/train.py --model yolo11n.pt --epochs 50
    python scripts/train.py --resume runs/train_12-06_14h05/weights/last.pt
"""

import argparse
import shutil
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

from compost_detection.curves import copy_results_preserving_history
from compost_detection.naming import run_name


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/train_default.yaml",
                        help="YAML d'hyperparamètres (défaut : configs/train_default.yaml)")
    parser.add_argument("--data", default="data/processed/data.yaml",
                        help="data.yaml généré par prepare_dataset.py")
    parser.add_argument("--runs-dir", default="runs",
                        help="dossier de sortie des runs (ex. /content/runs sur Colab)")
    parser.add_argument("--model", help="surcharge le modèle (yolov8n.pt, yolo11n.pt...)")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lr0", type=float,
                        help="learning rate initial, surcharge le YAML. Implique un optimiseur "
                             "explicite (AdamW si le YAML n'en fixe pas) : avec "
                             "'optimizer: auto', Ultralytics IGNORE lr0")
    parser.add_argument("--device", help="cpu, 0, 0,1... (défaut : auto-détecté)")
    parser.add_argument("--workers", type=int, default=2,
                        help="processus de chargement des données (défaut : 2 ; le défaut "
                             "Ultralytics de 8 sature la RAM limitée de WSL)")
    parser.add_argument("--run-prefix", default="train",
                        help="préfixe du dossier de run : 'pretrain' (datasets externes), "
                             "'finetune' (captures réelles)... (défaut : train)")
    parser.add_argument("--resume", metavar="LAST_PT",
                        help="chemin du last.pt d'un run interrompu à reprendre")
    parser.add_argument("--backup-dir",
                        help="copie périodique des checkpoints vers ce dossier (ex. Drive monté)")
    parser.add_argument("--backup-every", type=int, default=10,
                        help="période de sauvegarde en epochs (défaut : 10)")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    for key in ("model", "epochs", "imgsz", "batch", "seed", "lr0"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    # 'optimizer: auto' (défaut Ultralytics) ignore lr0 et momentum et recalcule
    # un lr d'après le nombre de classes : un lr0 explicite exige un optimiseur
    # explicite, sinon le réglage n'est jamais appliqué.
    if "lr0" in cfg and str(cfg.get("optimizer", "auto")).lower() == "auto":
        cfg["optimizer"] = "AdamW"
        print(f"lr0={cfg['lr0']} fourni : optimizer forcé à AdamW ('auto' l'ignorerait)")
    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    if args.resume:
        model = YOLO(args.resume)
    else:
        model = YOLO(cfg["model"])

    if args.backup_dir:
        def backup_checkpoints(trainer):
            if (trainer.epoch + 1) % args.backup_every == 0:
                dst = Path(args.backup_dir) / trainer.save_dir.name
                shutil.copytree(trainer.save_dir / "weights", dst / "weights",
                                dirs_exist_ok=True)
                # courbes incluses : sans elles, une VM Colab recyclée les emporte
                # (après un --resume, l'historique est archivé, jamais écrasé)
                copy_results_preserving_history(trainer.save_dir, dst)
                print(f"Checkpoints sauvegardés vers {dst}")
        model.add_callback("on_fit_epoch_end", backup_checkpoints)

    if args.resume:
        model.train(resume=True)
    else:
        # tout le YAML (sauf 'model') part tel quel à Ultralytics
        hyp = {k: v for k, v in cfg.items() if k != "model"}
        model.train(
            data=str(Path(args.data).resolve()),
            workers=args.workers,
            device=device,
            # chemin absolu : sinon Ultralytics le préfixe par son runs_dir global
            project=str(Path(args.runs_dir).resolve()),
            name=run_name(args.run_prefix),
            **hyp,
        )

    save_dir = Path(model.trainer.save_dir)
    used = {**cfg, "data": args.data, "device": device}
    with open(save_dir / "train_config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(used, f, allow_unicode=True)

    best = save_dir / "weights" / "best.pt"
    print(f"\nEntraînement terminé. Meilleur modèle : {best}")
    print("Pour déployer vers l'interface d'annotation :")
    print(f"  cp {best} ../weights/best.pt")


if __name__ == "__main__":
    main()
