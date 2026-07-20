"""Importe un dataset externe vers le format attendu par prepare_dataset.py.

Copie les images vers <output>/images/, réécrit les labels au format YOLO
avec les ids de classes de configs/data.yaml (table de correspondance dans un
YAML, voir configs/import_example.yaml), et génère <output>/groups.csv qui
fixe la granularité du split (par image ou par dossier source).

Formats de labels source implémentés : yolo (un .txt par image, lignes
``class_id cx cy w h`` normalisées) et yolo-seg (polygones de segmentation
``class_id x1 y1 x2 y2 ...``, convertis en boîtes englobantes). Pour un autre
format (COCO json, VOC xml...), écrire une fonction de même signature que
parse_yolo et l'ajouter dans PARSERS.

Exemple :
    python scripts/import_dataset.py --source ~/datasets/trashnet \\
        --mapping configs/import_trashnet.yaml --output data/raw/trashnet
    python scripts/prepare_dataset.py --source data/raw/trashnet
"""

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def find_yolo_label(image_path, source):
    """Cherche le .txt d'une image : à côté d'elle, ou dans une arborescence
    labels/ parallèle au dossier images/ (layout Ultralytics classique)."""
    candidates = [image_path.with_suffix(".txt")]
    parts = image_path.relative_to(source).with_suffix(".txt").parts
    for i, part in enumerate(parts):
        if part == "images":
            candidates.append(source.joinpath(*parts[:i], "labels", *parts[i + 1:]))
    return next((p for p in candidates if p.exists()), None)


def parse_yolo(image_path, source, source_names):
    """Annotations d'une image dont les labels source sont déjà au format YOLO.

    Retourne une liste de (nom_de_classe_source, [cx, cy, w, h] en str),
    ou None si l'image n'a pas de fichier de label.
    """
    label_path = find_yolo_label(image_path, source)
    if label_path is None:
        return None
    annotations = []
    for line in label_path.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise SystemExit(f"{label_path}: ligne invalide '{line}' "
                             "(attendu : class_id cx cy w h)")
        class_id = int(fields[0])
        if class_id >= len(source_names):
            raise SystemExit(f"{label_path}: id de classe {class_id} hors de "
                             f"source_names ({len(source_names)} classes déclarées)")
        annotations.append((source_names[class_id], fields[1:]))
    return annotations


def parse_yolo_seg(image_path, source, source_names):
    """Annotations au format YOLO segmentation : ``class x1 y1 x2 y2 ...``.

    Chaque polygone est converti en sa boîte englobante YOLO (cx cy w h).
    Les lignes à 5 champs (déjà des boîtes) sont acceptées telles quelles.
    """
    label_path = find_yolo_label(image_path, source)
    if label_path is None:
        return None
    annotations = []
    for line in label_path.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        class_id = int(fields[0])
        if class_id >= len(source_names):
            raise SystemExit(f"{label_path}: id de classe {class_id} hors de "
                             f"source_names ({len(source_names)} classes déclarées)")
        if len(fields) == 5:
            annotations.append((source_names[class_id], fields[1:]))
            continue
        coords = [float(v) for v in fields[1:]]
        if len(coords) < 6 or len(coords) % 2:
            raise SystemExit(f"{label_path}: ligne invalide '{line[:60]}...' "
                             "(attendu : class_id suivi de paires x y du polygone)")
        xs, ys = coords[0::2], coords[1::2]
        x_min, x_max = max(min(xs), 0.0), min(max(xs), 1.0)
        y_min, y_max = max(min(ys), 0.0), min(max(ys), 1.0)
        box = [(x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min]
        annotations.append((source_names[class_id], [f"{v:.6f}" for v in box]))
    return annotations


PARSERS = {"yolo": parse_yolo, "yolo-seg": parse_yolo_seg}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="racine du dataset externe")
    parser.add_argument("--mapping", required=True,
                        help="YAML de correspondance (voir configs/import_example.yaml)")
    parser.add_argument("--output", required=True,
                        help="dossier de sortie, ex. data/raw/trashnet")
    parser.add_argument("--classes", default="configs/data.yaml",
                        help="YAML définissant les classes cibles (défaut : configs/data.yaml)")
    parser.add_argument("--group-by", choices=("image", "folder"), default="image",
                        help="granularité du split dans groups.csv : 'image' (images "
                             "indépendantes) ou 'folder' (dossier parent = groupe, si le "
                             "dataset est organisé par scène/batch)")
    args = parser.parse_args()

    source = Path(args.source).expanduser()
    output = Path(args.output)
    mapping_cfg = yaml.safe_load(open(args.mapping, encoding="utf-8"))
    fmt = mapping_cfg["format"]
    if fmt not in PARSERS:
        raise SystemExit(f"Format '{fmt}' non implémenté (disponibles : "
                         f"{', '.join(PARSERS)}). Ajouter une fonction dans PARSERS.")
    source_names = mapping_cfg["source_names"]
    class_map = mapping_cfg["class_map"] or {}
    target_names = yaml.safe_load(open(args.classes, encoding="utf-8"))["names"]
    target_id = {name: i for i, name in enumerate(target_names)}
    for src_name, dst_name in class_map.items():
        if dst_name is not None and dst_name not in target_id:
            parser.error(f"class_map: '{src_name}' -> '{dst_name}' inconnu de "
                         f"{args.classes} (classes : {', '.join(target_names)})")

    images = sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"Aucune image trouvée sous {source}")

    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "labels").mkdir(parents=True, exist_ok=True)
    kept = Counter()       # instances gardées, par classe cible
    dropped = Counter()    # instances ignorées, par classe source
    n_negatives = 0        # images sans aucune annotation gardée
    n_unlabeled = 0        # images sans fichier de label du tout
    rows = []              # (stem, group_id) pour groups.csv
    seen_stems = set()
    for image_path in images:
        rel = image_path.relative_to(source)
        # nom de sortie unique : dataset + chemin relatif aplati
        # (a/b/img.jpg -> <dataset>_a_b_img.jpg — évite les collisions entre
        # datasets quand plusieurs sources s'accumulent dans data/processed)
        stem = "_".join([output.name, *rel.with_suffix("").parts]).replace(" ", "_")
        if stem in seen_stems:
            raise SystemExit(f"Collision de noms après aplatissement : '{stem}' "
                             f"(en traitant {rel}) — renommer les fichiers concernés.")
        seen_stems.add(stem)

        annotations = PARSERS[fmt](image_path, source, source_names)
        if annotations is None:
            n_unlabeled += 1
            annotations = []
        lines = []
        for src_name, coords in annotations:
            dst_name = class_map.get(src_name)
            if dst_name is None:
                dropped[src_name] += 1
                continue
            kept[dst_name] += 1
            lines.append(" ".join([str(target_id[dst_name]), *coords]))

        shutil.copy2(image_path, output / "images" / f"{stem}{image_path.suffix.lower()}")
        if lines:
            (output / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        else:
            n_negatives += 1
        # group_id préfixé par le nom du dataset (déjà inclus dans stem) pour que
        # le hash de split ne collisionne pas avec les groupes d'un autre import
        if args.group_by == "image":
            group_id = stem
        else:
            group_id = f"{output.name}_{rel.parent.as_posix() or source.name}"
        rows.append((stem, group_id))

    with open(output / "groups.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stem", "group_id"])
        writer.writerows(rows)

    print(f"Importé : {len(images)} images -> {output}")
    print(f"  groupes ({args.group_by}) : {len(set(g for _, g in rows))}")
    print(f"  images négatives (aucune annotation gardée) : {n_negatives}, "
          f"dont sans fichier de label : {n_unlabeled}")
    print("  instances gardées par classe cible :")
    for name in target_names:
        print(f"    {name}: {kept[name]}")
    if dropped:
        print("  instances ignorées par classe source (null / absente de class_map) :")
        for name, count in dropped.most_common():
            print(f"    {name}: {count}")
    print(f"\nÉtape suivante : python scripts/prepare_dataset.py --source {output}")


if __name__ == "__main__":
    main()
