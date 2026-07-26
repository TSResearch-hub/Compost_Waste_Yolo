"""Tests des parsers de labels du script d'import (scripts/import_dataset.py)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from import_dataset import find_yolo_label, parse_yolo, parse_yolo_seg  # noqa: E402

NAMES = ["plastic", "cardboard"]


def make_capture(tmp_path, label_text, layout="flat"):
    """Crée une image factice et son label selon le layout demandé."""
    if layout == "flat":
        image = tmp_path / "img.jpg"
        label = tmp_path / "img.txt"
    else:  # arborescence images/ + labels/ parallèles
        image = tmp_path / "train" / "images" / "img.jpg"
        label = tmp_path / "train" / "labels" / "img.txt"
    image.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    image.write_text("fake")
    label.write_text(label_text)
    return image


def test_parse_yolo_reads_boxes(tmp_path):
    image = make_capture(tmp_path, "1 0.5 0.5 0.2 0.1\n")
    assert parse_yolo(image, tmp_path, NAMES) == [("cardboard", ["0.5", "0.5", "0.2", "0.1"])]


def test_parse_yolo_missing_label_returns_none(tmp_path):
    image = tmp_path / "img.jpg"
    image.write_text("fake")
    assert parse_yolo(image, tmp_path, NAMES) is None


def test_parse_yolo_rejects_polygons(tmp_path):
    image = make_capture(tmp_path, "0 0.1 0.1 0.9 0.1 0.9 0.9\n")
    with pytest.raises(SystemExit, match="ligne invalide"):
        parse_yolo(image, tmp_path, NAMES)


def test_find_label_in_parallel_labels_dir(tmp_path):
    image = make_capture(tmp_path, "0 0.5 0.5 0.2 0.1\n", layout="parallel")
    assert find_yolo_label(image, tmp_path) == tmp_path / "train" / "labels" / "img.txt"


def test_seg_polygon_converted_to_bounding_box(tmp_path):
    # triangle : x dans [0.2, 0.6], y dans [0.1, 0.5]
    image = make_capture(tmp_path, "0 0.2 0.1 0.6 0.1 0.4 0.5\n")
    [(name, box)] = parse_yolo_seg(image, tmp_path, NAMES)
    assert name == "plastic"
    cx, cy, w, h = map(float, box)
    assert (cx, cy) == pytest.approx((0.4, 0.3))
    assert (w, h) == pytest.approx((0.4, 0.4))


def test_seg_accepts_plain_boxes_untouched(tmp_path):
    image = make_capture(tmp_path, "1 0.5 0.5 0.2 0.1\n")
    assert parse_yolo_seg(image, tmp_path, NAMES) == [("cardboard", ["0.5", "0.5", "0.2", "0.1"])]


def test_seg_clamps_out_of_range_coordinates(tmp_path):
    image = make_capture(tmp_path, "0 -0.1 0.2 1.2 0.2 0.5 0.8\n")
    [(_, box)] = parse_yolo_seg(image, tmp_path, NAMES)
    cx, cy, w, h = map(float, box)
    assert 0 <= cx - w / 2 and cx + w / 2 <= 1
    assert 0 <= cy - h / 2 and cy + h / 2 <= 1


def test_seg_rejects_odd_coordinate_count(tmp_path):
    image = make_capture(tmp_path, "0 0.1 0.1 0.9 0.1 0.9\n")
    with pytest.raises(SystemExit, match="paires"):
        parse_yolo_seg(image, tmp_path, NAMES)


@pytest.mark.parametrize("parser", [parse_yolo, parse_yolo_seg])
def test_unknown_class_id_raises(tmp_path, parser):
    image = make_capture(tmp_path, "7 0.5 0.5 0.2 0.1\n")
    with pytest.raises(SystemExit, match="hors de"):
        parser(image, tmp_path, NAMES)
