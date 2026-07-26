"""Prépare le dataset : reconstruction des groupes puis split PAR GROUPE.

Lit un dossier d'images + labels YOLO (à plat ou dans des sous-dossiers
images/ et labels/), regroupe les images (groups.csv prioritaire, puis
sessions cap_{timestamp} via manifeste sessions.csv ou clustering temporel,
sinon un groupe par image) et copie les fichiers vers le format attendu par
Ultralytics. Toutes les images d'un même groupe vont dans le même split.

Sources acceptées : export de l'interface d'annotation (cap_{timestamp}.jpg)
ou dataset externe converti par scripts/import_dataset.py (noms libres).

Exemple :
    python scripts/prepare_dataset.py --source data/raw/export_2026_06
"""

import argparse
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from compost_detection.splitting import (
    assign_split,
    build_session_mapping,
    try_parse_capture_timestamp,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True,
                        help="export de l'interface d'annotation ou sortie de import_dataset.py")
    parser.add_argument("--output", default="data/processed",
                        help="dossier de sortie au format Ultralytics (défaut : data/processed)")
    parser.add_argument("--classes", default="configs/data.yaml",
                        help="YAML définissant les classes (défaut : configs/data.yaml)")
    parser.add_argument("--ratios", nargs=3, type=float, default=[0.70, 0.15, 0.15],
                        metavar=("TRAIN", "VAL", "TEST"),
                        help="proportions train/val/test (défaut : 0.70 0.15 0.15)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed du hash d'affectation des sessions (défaut : 0)")
    parser.add_argument("--session-gap-minutes", type=float, default=60,
                        help="écart max entre deux captures d'une même session (défaut : 60)")
    parser.add_argument("--symlink", action="store_true",
                        help="créer des liens symboliques au lieu de copier")
    args = parser.parse_args()

    if abs(sum(args.ratios) - 1.0) > 1e-6:
        parser.error(f"--ratios doit sommer à 1 (reçu : {args.ratios})")

    source = Path(args.source)
    output = Path(args.output)
    class_names = yaml.safe_load(open(args.classes, encoding="utf-8"))["names"]

    # Recensement des captures (images à plat ou dans images/, labels à côté ou dans labels/)
    images = sorted(
        p for d in (source, source / "images") if d.is_dir()
        for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise SystemExit(f"Aucune image trouvée dans {source} (ni dans {source / 'images'})")
    captures = []  # (stem, image_path, label_path ou None)
    for image_path in images:
        label_path = next(
            (p for p in (image_path.with_suffix(".txt"),
                         source / "labels" / f"{image_path.stem}.txt") if p.exists()),
            None,
        )
        captures.append((image_path.stem, image_path, label_path))

    # Reconstruction des groupes puis affectation de chaque groupe à un split
    session_of, method = build_session_mapping(
        source, [stem for stem, _, _ in captures], args.session_gap_minutes)
    print(f"Regroupement des images : {method}")
    by_session = defaultdict(list)
    for capture in captures:
        by_session[session_of[capture[0]]].append(capture)
    split_of = {sid: assign_split(sid, tuple(args.ratios), args.seed) for sid in by_session}
    singletons = Counter()  # groupes d'une seule image, agrégés à l'affichage
    for sid in sorted(by_session):
        stems = [stem for stem, _, _ in by_session[sid]]
        if len(stems) == 1:
            singletons[split_of[sid]] += 1
            continue
        timestamps = [try_parse_capture_timestamp(stem) for stem in stems]
        if None not in timestamps:
            fmt = "%Y-%m-%d %H:%M"
            start = datetime.fromtimestamp(min(timestamps), tz=timezone.utc).strftime(fmt)
            end = datetime.fromtimestamp(max(timestamps), tz=timezone.utc).strftime(fmt)
            print(f"  {sid}: {start} -> {end} UTC, {len(stems)} images -> {split_of[sid]}")
        else:
            print(f"  {sid}: {len(stems)} images -> {split_of[sid]}")
    if singletons:
        detail = ", ".join(f"{n} -> {split}" for split, n in sorted(singletons.items()))
        print(f"  + {sum(singletons.values())} groupes d'une seule image ({detail})")
    if "clustering temporel" in method and len(by_session) == 1 and len(captures) > 1:
        print("ATTENTION : le clustering n'a produit qu'UNE seule session — "
              "--session-gap-minutes est peut-être trop grand pour ces données.")

    # Copie (ou symlink) vers la structure Ultralytics
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    transfer = (lambda src, dst: dst.symlink_to(src.resolve())) if args.symlink else shutil.copy2
    instances = {s: Counter() for s in ("train", "val", "test")}
    n_images = Counter()
    n_negatives = Counter()
    for sid, session_captures in by_session.items():
        split = split_of[sid]
        for _, image_path, label_path in session_captures:
            dst_image = output / "images" / split / image_path.name
            if not dst_image.exists():
                transfer(image_path, dst_image)
            n_images[split] += 1
            lines = label_path.read_text().split() if label_path else []
            if label_path and lines:
                dst_label = output / "labels" / split / label_path.name
                if not dst_label.exists():
                    transfer(label_path, dst_label)
                for class_id in lines[::5]:  # 1 ligne = class_id + 4 coordonnées
                    instances[split][class_names[int(class_id)]] += 1
            else:
                n_negatives[split] += 1

    # data.yaml consommé par train.py / evaluate.py (chemin absolu : pas de
    # dépendance au datasets_dir global d'Ultralytics)
    data_yaml = {
        "path": str(output.resolve()),
        "train": "images/train", "val": "images/val", "test": "images/test",
        "names": dict(enumerate(class_names)),
    }
    with open(output / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)

    # Récapitulatif
    print("\nRécapitulatif par split :")
    sessions_per_split = Counter(split_of.values())
    for split in ("train", "val", "test"):
        print(f"  {split}: {sessions_per_split[split]} sessions, {n_images[split]} images "
              f"(dont {n_negatives[split]} négatives, sans aucun label)")
        for name in class_names:
            print(f"    {name}: {instances[split][name]} instances")
        if n_images[split] == 0:
            print(f"  ATTENTION : le split '{split}' est vide.")
    for name in class_names:
        missing = [s for s in ("train", "val", "test")
                   if n_images[s] > 0 and instances[s][name] == 0]
        if missing:
            print(f"ATTENTION : la classe '{name}' est absente de : {', '.join(missing)}")
    print(f"\nDataset prêt : {output / 'data.yaml'}")


if __name__ == "__main__":
    main()
