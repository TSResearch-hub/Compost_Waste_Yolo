"""Split des captures réelles pour le fine-tuning + l'éval « compost ».

Découpe les captures (sortie de import_dataset.py, avec images/ + labels/) en deux :

  - un POOL de FINE-TUNING (images + labels bruts) -> à passer à prepare_dataset.py ;
  - un TEST COMPOST mis de côté (avec data.yaml prêt) -> à passer à evaluate.py.

Le test est la SEULE mesure honnête en conditions réelles : ces images ne sont
jamais vues à l'entraînement. On évalue dessus le modèle pré-entraîné (éval B)
PUIS le modèle fine-tuné (éval C) pour voir ce que le fine-tuning apporte.

Split STRATIFIÉ par session (groups.csv fait foi, sinon préfixe du nom) :
--test-fraction de CHAQUE session part au test -> les deux sessions sont
représentées des deux côtés, proportions conservées. Le test est structuré
images/test + labels/test (attendu par evaluate.py).

Le pool de fine-tuning est laissé SANS groups.csv : prepare_dataset.py y fera un
split train/val PAR IMAGE (avec 2 sessions seulement, un split par session ne
peut pas remplir train ET val). La contamination train/val n'a pas d'importance
ici : la seule mesure honnête est captures_test, entièrement mis de côté.

Usage :
    python scripts/split_captures.py --source data/raw/captures --output data/finetune

Options : --test-fraction 0.2  --seed 42

Puis (voir README, section « Fine-tuning ») :
    prepare_dataset.py sur le pool -> train.py (fine-tune) -> evaluate.py (B et C).
"""

import argparse
import csv
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
IMG_EXTS = (".jpg", ".jpeg", ".png")


def group_of(stem, groups):
    """Session/groupe d'une capture : groups.csv si présent, sinon préfixe du nom."""
    if stem in groups:
        return groups[stem]
    # fallback : captures_1_images_... / captures_1.5_images_... -> captures_1[.5]
    parts = stem.split("_images_", 1)
    return parts[0] if len(parts) == 2 else stem


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="data/raw/captures",
                    help="captures importées (dossier avec images/ et labels/)")
    ap.add_argument("--output", default="data/finetune",
                    help="dossier de sortie (défaut : data/finetune)")
    ap.add_argument("--test-fraction", type=float, default=0.2,
                    help="part de CHAQUE session mise au test compost (défaut : 0.2)")
    ap.add_argument("--seed", type=int, default=42, help="graine du tirage (défaut : 42)")
    args = ap.parse_args()

    random.seed(args.seed)
    src = Path(args.source)
    if not (src / "images").is_dir():
        sys.exit(f"{src}/images introuvable (attends une sortie de import_dataset.py)")

    # groups.csv (stem -> group_id) s'il existe
    groups = {}
    gcsv = src / "groups.csv"
    if gcsv.exists():
        with open(gcsv, encoding="utf-8") as f:
            groups = {row["stem"]: row["group_id"] for row in csv.DictReader(f)}

    # images par session, puis tirage stratifié
    by_group = {}
    for img in sorted((src / "images").iterdir()):
        if img.suffix.lower() in IMG_EXTS:
            by_group.setdefault(group_of(img.stem, groups), []).append(img)

    assign = {}  # img -> "finetune" | "test"
    for imgs in by_group.values():
        random.shuffle(imgs)
        n_test = round(len(imgs) * args.test_fraction)
        for i, img in enumerate(imgs):
            assign[img] = "test" if i < n_test else "finetune"

    out = Path(args.output)
    ft, te = out / "captures_finetune", out / "captures_test"
    for d in (ft, te):
        shutil.rmtree(d, ignore_errors=True)
    dest = {"finetune": (ft / "images", ft / "labels"),
            "test": (te / "images" / "test", te / "labels" / "test")}
    for img_dir, lbl_dir in dest.values():
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

    for img, sub in assign.items():
        img_dir, lbl_dir = dest[sub]
        shutil.copy(img, img_dir / img.name)
        lbl = src / "labels" / f"{img.stem}.txt"
        if lbl.exists():
            shutil.copy(lbl, lbl_dir / lbl.name)

    # data.yaml du test (train/val/test pointent tous sur le seul split présent)
    names = yaml.safe_load(open(ROOT / "configs/data.yaml", encoding="utf-8"))["names"]
    with open(te / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"path": str(te.resolve()),
                        "train": "images/test", "val": "images/test", "test": "images/test",
                        "names": dict(enumerate(names))},
                       f, allow_unicode=True, sort_keys=False)

    # récap
    counts = Counter((group_of(i.stem, groups), s) for i, s in assign.items())
    order = sorted(by_group)
    for sub in ("finetune", "test"):
        n_img = sum(1 for _ in dest[sub][0].glob("*"))
        n_lbl = sum(1 for _ in dest[sub][1].glob("*.txt"))
        detail = ", ".join(f"{g}:{counts[(g, sub)]}" for g in order)
        print(f"  {sub:9s}: {n_img} images, {n_lbl} labels  ({detail})")
    print(f"\nPool fine-tuning -> {ft}")
    print(f"Test compost     -> {te}/data.yaml")
    print("\nÉtapes suivantes (voir README) :")
    print(f"  python scripts/prepare_dataset.py --source {ft} "
          f"--output {out}/dataset_finetune --ratios 0.85 0.15 0")
    print(f"  python scripts/evaluate.py --weights <pretrain.pt> "
          f"--data {te}/data.yaml --split test          # éval B")
    print(f"  python scripts/train.py --model <pretrain.pt> "
          f"--data {out}/dataset_finetune/data.yaml --epochs 30 --lr0 0.001   # fine-tune")
    print(f"  python scripts/evaluate.py --weights runs/train_xxx/weights/best.pt "
          f"--data {te}/data.yaml --split test   # éval C")


if __name__ == "__main__":
    main()
