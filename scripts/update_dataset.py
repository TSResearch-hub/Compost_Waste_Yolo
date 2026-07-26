"""Met à jour le dataset de captures : crée un SNAPSHOT versionné et figé.

Chaque exécution produit un NOUVEAU dossier ``data/captures/vNNN_<date>``
(images/ + labels/ + groups.csv + md5.csv), construit à partir du snapshot
précédent + les annotations des sources. Les snapshots existants ne sont
JAMAIS modifiés : un entraînement référence toujours un dataset figé,
reproductible. Le lien ``data/captures/latest`` pointe vers le dernier.

Fusion des sources (dossiers `dataset_recolte`, un par poste d'annotation) :

- déduplication par CONTENU d'image (md5) : le même cliché récupéré de deux
  postes, ou déjà présent sous un autre nom, n'entre qu'une seule fois ;
- si une image déjà connue revient avec des annotations différentes, son label
  est mis à jour dans le NOUVEAU snapshot (la dernière annotation fait foi) ;
- chaque nouvelle image est rattachée à une session = date de prise de vue lue
  dans son nom (``WIN_20260714_...`` ou ``20260714_...``), via groups.csv.

Les images inchangées sont liées en dur (hardlink) vers le snapshot précédent :
un snapshot ne coûte que les nouveautés, pas une copie complète.

Le tout premier snapshot part de ``data/raw/captures`` s'il existe (l'ancien
dataset accumulé, laissé intact).

Usage :
    python scripts/update_dataset.py                       # source : dataset_recolte
    python scripts/update_dataset.py --source poste1/ poste2/ poste3/

Étape suivante : python scripts/retrain.py   (utilise data/captures/latest)
"""

import argparse
import csv
import hashlib
import os
import re
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def link_or_copy(src, dst):
    """Hardlink si possible (même disque, zéro octet), sinon copie."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def session_of(image_path):
    """Session d'une capture = date de prise de vue AAAAMMJJ lue dans le nom
    (WIN_20260714_... webcam, 20260714_... téléphone) ; sinon date du fichier."""
    m = re.search(r"(20\d{6})", image_path.stem)
    date = m.group(1) if m else f"{datetime.fromtimestamp(image_path.stat().st_mtime):%Y%m%d}"
    return f"session_{date}"


def map_label(label_path, source_names, class_map, target_id):
    """Convertit un label de l'interface vers les ids de data.yaml.

    Retourne (lignes converties, Counter des instances gardées par classe cible,
    Counter des instances ignorées par classe source).
    """
    lines, kept, dropped = [], Counter(), Counter()
    if label_path is None or not label_path.exists():
        return lines, kept, dropped
    for line in label_path.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        class_id = int(fields[0])
        if class_id >= len(source_names):
            sys.exit(f"{label_path}: id de classe {class_id} inconnu — la liste des "
                     f"classes de l'interface a-t-elle changé ? (voir import_captures.yaml)")
        src_name = source_names[class_id]
        dst_name = class_map.get(src_name)
        if dst_name is None:
            dropped[src_name] += 1
            continue
        kept[dst_name] += 1
        lines.append(" ".join([str(target_id[dst_name]), *fields[1:]]))
    return lines, kept, dropped


def source_files(source):
    """Couples (image, label) d'une source : layout images/+labels/ ou à plat."""
    img_dir = source / "images" if (source / "images").is_dir() else source
    lbl_dir = source / "labels" if (source / "labels").is_dir() else img_dir
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() in IMAGE_SUFFIXES:
            yield img, lbl_dir / f"{img.stem}.txt"


def find_previous(captures_root, seed):
    """Dernier snapshot existant, sinon le dataset d'amorçage (data/raw/captures)."""
    if captures_root.is_dir():
        snaps = sorted(d for d in captures_root.iterdir()
                       if d.is_dir() and re.match(r"v\d+", d.name))
        if snaps:
            return snaps[-1]
    return seed if seed.is_dir() else None


def load_md5_index(dataset):
    """Index md5 -> stem d'un dataset ; lit md5.csv s'il existe, sinon calcule."""
    idx_file = dataset / "md5.csv"
    if idx_file.exists():
        with open(idx_file, encoding="utf-8") as f:
            return {r["md5"]: r["stem"] for r in csv.DictReader(f)}
    print(f"  (pas de md5.csv dans {dataset.name}, calcul des empreintes...)")
    return {md5_of(p): p.stem for p in (dataset / "images").iterdir()
            if p.suffix.lower() in IMAGE_SUFFIXES}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", nargs="+", default=[str(ROOT / "dataset_recolte")],
                    help="dossier(s) d'annotations à intégrer (défaut : dataset_recolte)")
    ap.add_argument("--captures-root", default="data/captures",
                    help="dossier des snapshots versionnés (défaut : data/captures)")
    ap.add_argument("--seed", default="data/raw/captures",
                    help="dataset d'amorçage du tout premier snapshot")
    ap.add_argument("--mapping", default="configs/import_captures.yaml",
                    help="correspondance classes interface -> data.yaml")
    ap.add_argument("--classes", default="configs/data.yaml")
    args = ap.parse_args()

    mapping = yaml.safe_load(open(ROOT / args.mapping, encoding="utf-8"))
    source_names, class_map = mapping["source_names"], mapping["class_map"] or {}
    target_names = yaml.safe_load(open(ROOT / args.classes, encoding="utf-8"))["names"]
    target_id = {name: i for i, name in enumerate(target_names)}

    captures_root = ROOT / args.captures_root
    previous = find_previous(captures_root, ROOT / args.seed)

    # nouveau snapshot : vNNN_<jour-mois>
    n = 1 + max((int(m.group(1)) for d in captures_root.glob("v*")
                 if (m := re.match(r"v(\d+)", d.name))), default=0)
    snapshot = captures_root / f"v{n:03d}_{datetime.now():%d-%m}"
    img_out, lbl_out = snapshot / "images", snapshot / "labels"
    img_out.mkdir(parents=True)
    lbl_out.mkdir(parents=True)

    # 1. base = contenu du snapshot précédent (hardlinks, aucun octet copié)
    known, group_rows = {}, []
    if previous:
        print(f"Base : {previous}")
        known = load_md5_index(previous)
        for img in (previous / "images").iterdir():
            if img.suffix.lower() in IMAGE_SUFFIXES:
                link_or_copy(img, img_out / img.name)
        for lbl in (previous / "labels").glob("*.txt"):
            shutil.copy2(lbl, lbl_out / lbl.name)   # copie : les labels peuvent changer
        prev_groups = previous / "groups.csv"
        if prev_groups.exists():
            with open(prev_groups, encoding="utf-8") as f:
                group_rows = [(r["stem"], r["group_id"]) for r in csv.DictReader(f)]
    else:
        print("Aucun snapshot ni dataset d'amorçage : premier snapshot depuis les sources.")
    stems = set(known.values())

    # 2. intégration des sources dans le nouveau snapshot
    added, updated, unchanged, negatives = 0, 0, 0, 0
    kept_total, dropped_total = Counter(), Counter()
    for src in args.source:
        src = Path(src).expanduser()
        if not src.is_dir():
            sys.exit(f"Source introuvable : {src}")
        print(f"Source : {src}")
        for img, lbl in source_files(src):
            lines, kept, dropped = map_label(lbl, source_names, class_map, target_id)
            kept_total.update(kept)
            dropped_total.update(dropped)
            digest = md5_of(img)

            if digest in known:  # image déjà connue -> au plus une mise à jour du label
                stem = known[digest]
                existing = lbl_out / f"{stem}.txt"
                old = existing.read_text() if existing.exists() else ""
                new = "\n".join(lines) + "\n" if lines else ""
                if new != old:
                    existing.write_text(new) if new else existing.unlink(missing_ok=True)
                    updated += 1
                else:
                    unchanged += 1
                continue

            stem, i = img.stem, 2  # nouveau contenu : stem unique
            while stem in stems:
                stem = f"{img.stem}_{i}"
                i += 1
            stems.add(stem)
            known[digest] = stem
            link_or_copy(img, img_out / f"{stem}{img.suffix.lower()}")
            if lines:
                (lbl_out / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            else:
                negatives += 1
            group_rows.append((stem, session_of(img)))
            added += 1

    # 3. index et métadonnées du snapshot
    with open(snapshot / "groups.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stem", "group_id"])
        writer.writerows(group_rows)
    with open(snapshot / "md5.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["md5", "stem"])
        writer.writerows(sorted(known.items(), key=lambda kv: kv[1]))

    # lien « latest » -> ce snapshot
    latest = captures_root / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(snapshot.name)

    total = sum(1 for p in img_out.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    print(f"\nNouveau snapshot (figé) : {snapshot}")
    print(f"  nouvelles images        : {added} (dont {negatives} sans annotation)")
    print(f"  labels mis à jour       : {updated}")
    print(f"  déjà à jour (ignorées)  : {unchanged}")
    print(f"  total du snapshot       : {total} images")
    if kept_total:
        print("  instances vues par classe (sources) :")
        for name in target_names:
            if kept_total[name]:
                print(f"    {name}: {kept_total[name]}")
    if dropped_total:
        print("  instances ignorées (classes compostables) : "
              + ", ".join(f"{n}: {c}" for n, c in dropped_total.most_common()))
    if negatives:
        print(f"\n  ⚠️ {negatives} nouvelles images sans annotation : de vraies images "
              "« sans intrus », ou des images pas encore annotées ? À vérifier.")
    print(f"\n{captures_root}/latest -> {snapshot.name}")
    print("Étape suivante : python scripts/retrain.py")


if __name__ == "__main__":
    main()
