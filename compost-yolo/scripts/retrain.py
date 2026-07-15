"""Réentraîne le modèle sur le dataset de captures à jour, en une commande.

Enchaîne les étapes du fine-tuning (chacune reste un script utilisable seul) :

  1. split_captures.py     — met de côté le TEST compost (jamais entraîné)
  2. prepare_dataset.py    — train/val du pool de fine-tuning
  3. evaluate.py           — éval AVANT (le pré-entraîné, référence)
  4. train.py              — fine-tuning depuis le pré-entraîné (runs/finetune_*)
  5. evaluate.py           — éval APRÈS (même test) + comparaison affichée
  6. (--deploy)            — copie le best.pt vers l'interface d'annotation

On repart TOUJOURS du modèle pré-entraîné canonique (models/), jamais du
fine-tuné précédent : quand le dataset grandit, le split peut changer, et seul
un départ du pré-entraîné garantit que le test n'a jamais été appris.

Usage :
    python scripts/update_dataset.py     # d'abord, si nouvelles annotations
    python scripts/retrain.py
    python scripts/retrain.py --pretrain models/pretrain_rtdetr-l.pt --batch 4
    python scripts/retrain.py --deploy   # déploie vers ../weights/ à la fin
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd):
    print("\n$ " + " ".join(cmd))
    if subprocess.call([sys.executable] + cmd, cwd=ROOT) != 0:
        sys.exit(f"Échec de l'étape : {cmd[0]}")


def latest_best(runs_dir, prefix):
    runs = sorted((ROOT / runs_dir).glob(f"{prefix}_*/weights/best.pt"),
                  key=lambda p: p.stat().st_mtime)
    if not runs:
        sys.exit(f"Aucun {runs_dir}/{prefix}_*/weights/best.pt trouvé")
    return runs[-1]


def latest_eval_json(runs_dir, prefix):
    evals = sorted((ROOT / runs_dir).glob(f"{prefix}_*/image_level_metrics.json"),
                   key=lambda p: p.stat().st_mtime)
    return json.load(open(evals[-1], encoding="utf-8")) if evals else None


def fmt(x):
    return "-" if x is None else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pretrain", default="models/pretrain_yolov8n.pt",
                    help="modèle pré-entraîné de départ (défaut : models/pretrain_yolov8n.pt)")
    ap.add_argument("--captures", default=None,
                    help="dataset de captures : un snapshot précis (data/captures/vNNN_*) ; "
                         "défaut : data/captures/latest, sinon data/raw/captures")
    ap.add_argument("--workdir", default="data/finetune",
                    help="dossier de travail du split (défaut : data/finetune)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr0", type=float, default=0.001)
    ap.add_argument("--batch", type=int, default=8, help="baisser à 4 si mémoire GPU insuffisante")
    ap.add_argument("--device", help="cpu, 0... (défaut : auto)")
    ap.add_argument("--test-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--skip-eval-before", action="store_true",
                    help="saute l'éval du pré-entraîné (si déjà faite sur ce split)")
    ap.add_argument("--deploy", action="store_true",
                    help="copie le best.pt final vers ../weights/best.pt (interface)")
    args = ap.parse_args()

    pretrain = ROOT / args.pretrain
    if not pretrain.exists():
        sys.exit(f"Pré-entraîné introuvable : {pretrain}\n"
                 "Copier le best.pt du pré-entraînement vers models/ (voir README).")
    if args.captures is None:
        latest = ROOT / "data/captures/latest"
        args.captures = "data/captures/latest" if latest.exists() else "data/raw/captures"
    print(f"Dataset de captures : {args.captures}"
          + (f" -> {(ROOT / args.captures).resolve().name}"
             if (ROOT / args.captures).is_symlink() else ""))
    device = ["--device", args.device] if args.device else []
    workdir = args.workdir
    test_yaml = f"{workdir}/captures_test/data.yaml"

    # 1-2. split + préparation train/val
    run(["scripts/split_captures.py", "--source", args.captures, "--output", workdir,
         "--test-fraction", str(args.test_fraction), "--seed", str(args.seed)])
    run(["scripts/prepare_dataset.py", "--source", f"{workdir}/captures_finetune",
         "--output", f"{workdir}/dataset_finetune", "--ratios", "0.85", "0.15", "0",
         "--symlink"])   # liens symboliques : économise le disque, sources figées

    # 3. éval AVANT (référence : le pré-entraîné sur le test compost)
    if not args.skip_eval_before:
        run(["scripts/evaluate.py", "--weights", str(pretrain),
             "--data", test_yaml, "--split", "test", "--runs-dir", args.runs_dir] + device)
    before = latest_eval_json(args.runs_dir, "eval_pretrain")

    # 4. fine-tuning (repart du pré-entraîné, learning rate bas)
    run(["scripts/train.py", "--model", str(pretrain),
         "--data", f"{workdir}/dataset_finetune/data.yaml",
         "--epochs", str(args.epochs), "--lr0", str(args.lr0),
         "--batch", str(args.batch), "--run-prefix", "finetune",
         "--runs-dir", args.runs_dir] + device)
    best = latest_best(args.runs_dir, "finetune")

    # 5. éval APRÈS (même test)
    run(["scripts/evaluate.py", "--weights", str(best),
         "--data", test_yaml, "--split", "test", "--runs-dir", args.runs_dir] + device)
    after = latest_eval_json(args.runs_dir, "eval_finetune")

    print("\n" + "=" * 60)
    print("Comparaison sur le test compost (niveau image) :")
    print(f"  {'':24} {'avant':>8} {'après':>8}")
    for label, key in (("rappel image", "recall"), ("précision image", "precision")):
        b = before["image_level"][key] if before else None
        a = after["image_level"][key] if after else None
        print(f"  {label:<24} {fmt(b):>8} {fmt(a):>8}")
    print(f"\nModèle fine-tuné : {best}")
    print(f"Détails : {args.runs_dir}/eval_pretrain_* (avant) vs "
          f"{args.runs_dir}/eval_finetune_* (après)")

    # 6. déploiement vers l'interface d'annotation
    if args.deploy:
        target = ROOT.parent / "weights" / "best.pt"
        target.parent.mkdir(exist_ok=True)
        import shutil
        shutil.copy2(best, target)
        print(f"\nDéployé vers l'interface : {target}")
    else:
        print(f"\nPour déployer vers l'interface : python scripts/retrain.py --deploy,"
              f"\nou : cp {best} ../weights/best.pt")


if __name__ == "__main__":
    main()
