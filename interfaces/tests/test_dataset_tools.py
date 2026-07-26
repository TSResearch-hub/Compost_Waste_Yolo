"""Tests des outils d'analyse, de corbeille et d'export du dataset."""
import pytest

import dataset_tools as dt

CLASSES = ["Plastique", "Métal", "Carton", "Aluminium", "Céramique",
           "Verre", "Composite", "Éponge"]


def _write_pair(img_dir, lbl_dir, stem, label_lines, img_bytes=None):
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / f"{stem}.jpg").write_bytes(img_bytes if img_bytes is not None else stem.encode())
    (lbl_dir / f"{stem}.txt").write_text("\n".join(label_lines), encoding="utf-8")


# ── class_distribution ───────────────────────────────────────────────────────

def test_distribution(tmp_path):
    img, lbl = tmp_path / "images", tmp_path / "labels"
    _write_pair(img, lbl, "a", ["1 0.5 0.5 0.2 0.2", "1 0.3 0.3 0.1 0.1"])
    _write_pair(img, lbl, "b", ["0 0.5 0.5 0.2 0.2"])
    _write_pair(img, lbl, "vide", [])
    dist = dt.class_distribution(lbl, 9)
    assert dist["counts"][1] == 2
    assert dist["counts"][0] == 1
    assert dist["total_boxes"] == 3
    assert dist["n_labels"] == 3
    assert dist["n_empty"] == 1


def test_distribution_dossier_absent(tmp_path):
    dist = dt.class_distribution(tmp_path / "nexiste_pas", 9)
    assert dist["total_boxes"] == 0 and dist["n_labels"] == 0


# ── analyze_dataset ──────────────────────────────────────────────────────────

def test_anomalies(tmp_path):
    img, lbl = tmp_path / "images", tmp_path / "labels"
    _write_pair(img, lbl, "ok", ["1 0.5 0.5 0.2 0.2"])
    # Label orphelin (pas d'image)
    (lbl / "orphelin.txt").write_text("0 0.5 0.5 0.1 0.1")
    # Image sans label
    (img / "sans_label.jpg").write_bytes(b"xxx")
    # Ligne illisible + classe hors référentiel + bbox fantôme + hors cadre
    _write_pair(img, lbl, "cassee", [
        "pas une ligne yolo",
        "42 0.5 0.5 0.2 0.2",
        "1 0.5 0.5 0.001 0.001",
        "1 0.99 0.5 0.2 0.2",
    ])
    # Doublon de contenu sous deux noms
    _write_pair(img, lbl, "dup1", ["1 0.5 0.5 0.2 0.2"], img_bytes=b"MEME_CONTENU")
    _write_pair(img, lbl, "dup2", ["1 0.5 0.5 0.2 0.2"], img_bytes=b"MEME_CONTENU")

    r = dt.analyze_dataset(img, lbl, 9)
    assert r.orphan_labels == ["orphelin.txt"]
    assert r.unlabeled_images == ["sans_label.jpg"]
    assert [(f, ln) for f, ln, _ in r.invalid_lines] == [("cassee.txt", 1)]
    assert [(f, cid) for f, _, cid in r.out_of_range] == [("cassee.txt", 42)]
    assert len(r.degenerate_boxes) == 2  # fantôme + hors cadre
    assert r.duplicate_groups == [["dup1.jpg", "dup2.jpg"]]
    assert r.n_problems == 6


def test_dataset_sain(tmp_path):
    img, lbl = tmp_path / "images", tmp_path / "labels"
    _write_pair(img, lbl, "a", ["1 0.5 0.5 0.2 0.2"])
    r = dt.analyze_dataset(img, lbl, 9)
    assert r.n_problems == 0


# ── move_to_trash ────────────────────────────────────────────────────────────

def test_corbeille_deplace_image_et_label(tmp_path):
    img, lbl, trash = tmp_path / "images", tmp_path / "labels", tmp_path / "corbeille"
    _write_pair(img, lbl, "a", ["1 0.5 0.5 0.2 0.2"])
    dest = dt.move_to_trash(img, lbl, trash, "a.jpg")
    assert not (img / "a.jpg").exists() and not (lbl / "a.txt").exists()
    assert dest.exists() and (trash / "labels" / "a.txt").exists()


def test_corbeille_sans_label(tmp_path):
    img, lbl, trash = tmp_path / "images", tmp_path / "labels", tmp_path / "corbeille"
    img.mkdir(); lbl.mkdir()
    (img / "seule.jpg").write_bytes(b"x")
    dest = dt.move_to_trash(img, lbl, trash, "seule.jpg")
    assert dest.name == "seule.jpg" and dest.exists()


def test_corbeille_collision_suffixe(tmp_path):
    img, lbl, trash = tmp_path / "images", tmp_path / "labels", tmp_path / "corbeille"
    _write_pair(img, lbl, "a", ["1 0.5 0.5 0.2 0.2"])
    dt.move_to_trash(img, lbl, trash, "a.jpg")
    _write_pair(img, lbl, "a", ["0 0.5 0.5 0.2 0.2"])  # même nom, re-supprimé
    dest2 = dt.move_to_trash(img, lbl, trash, "a.jpg")
    assert dest2.name != "a.jpg"  # suffixe horodaté, pas d'écrasement


def test_corbeille_image_absente(tmp_path):
    with pytest.raises(FileNotFoundError):
        dt.move_to_trash(tmp_path, tmp_path, tmp_path / "t", "fantome.jpg")


# ── export_dataset ───────────────────────────────────────────────────────────

def _build_export_dataset(tmp_path, n_commun=20, n_rare=3):
    """n_commun images de Métal (1) + n_rare images de Carton (2)."""
    img, lbl = tmp_path / "images", tmp_path / "labels"
    for i in range(n_commun):
        _write_pair(img, lbl, f"metal_{i:02d}", ["1 0.5 0.5 0.2 0.2"])
    for i in range(n_rare):
        _write_pair(img, lbl, f"carton_{i}", ["2 0.5 0.5 0.2 0.2"])
    return img, lbl


def test_export_split_stratifie_preserve_classe_rare(tmp_path):
    img, lbl = _build_export_dataset(tmp_path)
    out = dt.export_dataset(img, lbl, tmp_path / "exports", CLASSES, val_ratio=0.2, seed=42)
    assert out["n_train"] + out["n_val"] == 23
    # La classe rare (3 images) doit exister des deux côtés du split
    assert out["per_class"]["Carton"][0] >= 1  # train
    assert out["per_class"]["Carton"][1] >= 1  # val
    # Arborescence YOLO + data.yaml
    assert len(list((out["out_dir"] / "images" / "train").iterdir())) == out["n_train"]
    assert len(list((out["out_dir"] / "labels" / "val").iterdir())) == out["n_val"]
    yaml_text = out["yaml_path"].read_text(encoding="utf-8")
    assert "nc: 8" in yaml_text
    assert "4: Céramique" in yaml_text  # accents préservés


def test_export_deterministe(tmp_path):
    img, lbl = _build_export_dataset(tmp_path)
    out1 = dt.export_dataset(img, lbl, tmp_path / "e1", CLASSES, seed=42)
    out2 = dt.export_dataset(img, lbl, tmp_path / "e2", CLASSES, seed=42)
    names1 = sorted(p.name for p in (out1["out_dir"] / "images" / "val").iterdir())
    names2 = sorted(p.name for p in (out2["out_dir"] / "images" / "val").iterdir())
    assert names1 == names2


def test_export_ignore_orphelins_et_sans_label(tmp_path):
    img, lbl = tmp_path / "images", tmp_path / "labels"
    _write_pair(img, lbl, "ok", ["1 0.5 0.5 0.2 0.2"])
    (img / "sans_label.jpg").write_bytes(b"x")
    (lbl / "orphelin.txt").write_text("0 0.5 0.5 0.1 0.1")
    out = dt.export_dataset(img, lbl, tmp_path / "exports", CLASSES)
    assert out["n_train"] + out["n_val"] == 1


def test_export_vide_leve_erreur(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    with pytest.raises(ValueError):
        dt.export_dataset(tmp_path / "images", tmp_path / "labels", tmp_path / "e", CLASSES)
