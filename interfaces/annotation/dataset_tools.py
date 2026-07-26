"""
Outils d'analyse et d'export du dataset YOLO (dataset_recolte/).

Module volontairement sans dépendance Streamlit / ultralytics : toutes les
fonctions sont pures (chemins + listes en entrée, structures en sortie), ce
qui permet de les tester et de les réutiliser (app PC, serveur mobile, CLI).

Trois familles de fonctions :
- analyse    : class_distribution, label_classes, analyze_dataset
- entretien  : move_to_trash (suppression réversible vers une corbeille)
- export     : export_dataset (split train/val stratifié + data.yaml)
"""
from __future__ import annotations

import hashlib
import random
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# En-dessous de cette taille normalisée (largeur OU hauteur), une bbox est
# considérée dégénérée : ~5 px sur une frame 1280, ~12 px sur une photo 3000.
# C'est la signature des « bboxes fantômes » créées par l'ancien bug de clic.
MIN_BOX_NORM = 0.004


def list_images(img_dir: Path) -> list[Path]:
    if not img_dir.exists():
        return []
    return sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS)


def parse_label_line(line: str):
    """Ligne YOLO « cid xc yc w h » → tuple (cid, xc, yc, w, h) ou None si invalide."""
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        cid = int(parts[0])
        xc, yc, w, h = (float(p) for p in parts[1:])
    except ValueError:
        return None
    return cid, xc, yc, w, h


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSE
# ══════════════════════════════════════════════════════════════════════════════

def class_distribution(lbl_dir: Path, n_classes: int) -> dict:
    """Compte les bboxes par classe sur tous les .txt de lbl_dir.

    Retourne {"counts": [n par classe], "total_boxes", "n_labels", "n_empty"}.
    Les lignes invalides ou hors plage sont ignorées ici (voir analyze_dataset
    pour leur inventaire détaillé).
    """
    counts = [0] * n_classes
    n_labels = n_empty = 0
    if lbl_dir.exists():
        for txt in sorted(lbl_dir.glob("*.txt")):
            n_labels += 1
            boxes = 0
            for line in txt.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                parsed = parse_label_line(line)
                if parsed and 0 <= parsed[0] < n_classes:
                    counts[parsed[0]] += 1
                    boxes += 1
            if boxes == 0:
                n_empty += 1
    return {
        "counts": counts,
        "total_boxes": sum(counts),
        "n_labels": n_labels,
        "n_empty": n_empty,
    }


def label_classes(lbl_dir: Path) -> dict[str, set[int]]:
    """stem → ensemble des classes présentes dans son label (pour filtres UI)."""
    result: dict[str, set[int]] = {}
    if lbl_dir.exists():
        for txt in lbl_dir.glob("*.txt"):
            classes = set()
            for line in txt.read_text(encoding="utf-8").splitlines():
                parsed = parse_label_line(line)
                if parsed:
                    classes.add(parsed[0])
            result[txt.stem] = classes
    return result


@dataclass
class DatasetReport:
    n_images: int = 0
    n_labels: int = 0
    n_empty_labels: int = 0
    unlabeled_images: list[str] = field(default_factory=list)   # image sans .txt
    orphan_labels: list[str] = field(default_factory=list)      # .txt sans image
    invalid_lines: list[tuple[str, int, str]] = field(default_factory=list)     # (fichier, n° ligne, contenu)
    out_of_range: list[tuple[str, int, int]] = field(default_factory=list)      # (fichier, n° ligne, class_id)
    degenerate_boxes: list[tuple[str, int, str]] = field(default_factory=list)  # (fichier, n° ligne, détail)
    duplicate_groups: list[list[str]] = field(default_factory=list)             # images au contenu identique

    @property
    def n_problems(self) -> int:
        return (
            len(self.orphan_labels) + len(self.invalid_lines)
            + len(self.out_of_range) + len(self.degenerate_boxes)
            + len(self.duplicate_groups)
        )


def analyze_dataset(img_dir: Path, lbl_dir: Path, n_classes: int,
                    check_duplicates: bool = True) -> DatasetReport:
    """Inventaire complet des anomalies du dataset.

    check_duplicates hache le contenu de toutes les images (quelques secondes
    sur un gros dataset) : désactivable pour une analyse rapide.
    """
    report = DatasetReport()
    images = list_images(img_dir)
    image_stems = {p.stem for p in images}
    report.n_images = len(images)

    label_files = sorted(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []
    label_stems = {p.stem for p in label_files}
    report.n_labels = len(label_files)

    report.unlabeled_images = sorted(p.name for p in images if p.stem not in label_stems)
    report.orphan_labels = sorted(p.name for p in label_files if p.stem not in image_stems)

    for txt in label_files:
        n_boxes = 0
        for lineno, line in enumerate(txt.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            parsed = parse_label_line(line)
            if parsed is None:
                report.invalid_lines.append((txt.name, lineno, line.strip()[:80]))
                continue
            cid, xc, yc, w, h = parsed
            n_boxes += 1
            if not 0 <= cid < n_classes:
                report.out_of_range.append((txt.name, lineno, cid))
            if w <= 0 or h <= 0 or min(w, h) < MIN_BOX_NORM:
                report.degenerate_boxes.append(
                    (txt.name, lineno, f"taille {w:.4f}×{h:.4f}")
                )
            elif (xc - w / 2 < -0.01 or yc - h / 2 < -0.01
                  or xc + w / 2 > 1.01 or yc + h / 2 > 1.01):
                report.degenerate_boxes.append(
                    (txt.name, lineno, f"hors cadre (centre {xc:.2f},{yc:.2f})")
                )
        if n_boxes == 0:
            report.n_empty_labels += 1

    if check_duplicates:
        by_hash: dict[str, list[str]] = {}
        for p in images:
            digest = hashlib.md5(p.read_bytes()).hexdigest()
            by_hash.setdefault(digest, []).append(p.name)
        report.duplicate_groups = sorted(
            names for names in by_hash.values() if len(names) > 1
        )

    return report


# ══════════════════════════════════════════════════════════════════════════════
# ENTRETIEN — corbeille
# ══════════════════════════════════════════════════════════════════════════════

def move_to_trash(img_dir: Path, lbl_dir: Path, trash_dir: Path, image_name: str) -> Path:
    """Déplace une image (et son label s'il existe) vers la corbeille.

    Suppression réversible : rien n'est effacé, tout part dans
    trash_dir/images et trash_dir/labels. En cas de collision de nom dans la
    corbeille, un suffixe horodaté est ajouté. Retourne le chemin de l'image
    déplacée.
    """
    src_img = img_dir / image_name
    if not src_img.exists():
        raise FileNotFoundError(f"Image introuvable : {src_img}")

    trash_img_dir = trash_dir / "images"
    trash_lbl_dir = trash_dir / "labels"
    trash_img_dir.mkdir(parents=True, exist_ok=True)
    trash_lbl_dir.mkdir(parents=True, exist_ok=True)

    stem, ext = src_img.stem, src_img.suffix
    dest_stem = stem
    if (trash_img_dir / f"{dest_stem}{ext}").exists():
        dest_stem = f"{stem}_{int(time.time())}"

    dest_img = trash_img_dir / f"{dest_stem}{ext}"
    shutil.move(str(src_img), str(dest_img))

    src_lbl = lbl_dir / f"{stem}.txt"
    if src_lbl.exists():
        shutil.move(str(src_lbl), str(trash_lbl_dir / f"{dest_stem}.txt"))

    return dest_img


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT — split train/val stratifié + data.yaml
# ══════════════════════════════════════════════════════════════════════════════

def _stratified_split(stems_classes: dict[str, set[int]], global_counts: list[int],
                      val_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    """Split train/val qui préserve les classes rares.

    Chaque image est rattachée à la classe la plus rare qu'elle contient
    (les images au label vide forment leur propre groupe « fond ») ; chaque
    groupe est ensuite réparti selon val_ratio. Un split purement aléatoire
    pouvait laisser une classe à 4 images totalement absente de la validation.
    """
    groups: dict[int, list[str]] = {}
    for stem, classes in stems_classes.items():
        if classes:
            key = min(classes, key=lambda c: (global_counts[c] if c < len(global_counts) else 0, c))
        else:
            key = -1  # image de fond (label vide)
        groups.setdefault(key, []).append(stem)

    rng = random.Random(seed)
    train, val = [], []
    for key in sorted(groups):
        members = sorted(groups[key])
        rng.shuffle(members)
        n_val = round(len(members) * val_ratio)
        n_val = min(n_val, len(members) - 1)  # toujours ≥ 1 image en train
        val.extend(members[:n_val])
        train.extend(members[n_val:])
    return sorted(train), sorted(val)


def export_dataset(img_dir: Path, lbl_dir: Path, out_root: Path,
                   class_names: list[str], val_ratio: float = 0.2,
                   seed: int = 42) -> dict:
    """Exporte les images annotées vers une arborescence YOLO prête à entraîner.

    out_root/export_YYYYMMDD_HHMMSS/
        data.yaml
        images/train  images/val
        labels/train  labels/val

    Seules les paires image + label sont exportées (un label vide est une
    image de fond légitime). Retourne un résumé : chemins, effectifs par
    split et par classe.
    """
    n_classes = len(class_names)
    images = {p.stem: p for p in list_images(img_dir)}
    stems_classes = {
        stem: classes
        for stem, classes in label_classes(lbl_dir).items()
        if stem in images
    }
    if not stems_classes:
        raise ValueError("Aucune paire image + label à exporter.")

    dist = class_distribution(lbl_dir, n_classes)
    train_stems, val_stems = _stratified_split(
        stems_classes, dist["counts"], val_ratio, seed
    )

    out_dir = out_root / f"export_{datetime.now():%Y%m%d_%H%M%S}"
    per_class = {name: [0, 0] for name in class_names}
    for split, stems in (("train", train_stems), ("val", val_stems)):
        split_img = out_dir / "images" / split
        split_lbl = out_dir / "labels" / split
        split_img.mkdir(parents=True, exist_ok=True)
        split_lbl.mkdir(parents=True, exist_ok=True)
        col = 0 if split == "train" else 1
        for stem in stems:
            shutil.copy2(images[stem], split_img / images[stem].name)
            shutil.copy2(lbl_dir / f"{stem}.txt", split_lbl / f"{stem}.txt")
            for cid in stems_classes[stem]:
                if 0 <= cid < n_classes:
                    per_class[class_names[cid]][col] += 1

    yaml_path = out_dir / "data.yaml"
    yaml_lines = [
        f"# Dataset Composte IA — exporté le {datetime.now():%Y-%m-%d %H:%M}",
        f"# {len(train_stems)} images train / {len(val_stems)} val (seed {seed})",
        f"path: {out_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {n_classes}",
        "names:",
    ] + [f"  {i}: {name}" for i, name in enumerate(class_names)]
    yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    return {
        "out_dir": out_dir,
        "yaml_path": yaml_path,
        "n_train": len(train_stems),
        "n_val": len(val_stems),
        "per_class": per_class,  # nom → [images train contenant la classe, val]
    }
