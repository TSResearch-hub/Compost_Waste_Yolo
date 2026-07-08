"""Tests du nommage des dossiers de runs."""

from datetime import datetime

from compost_detection.naming import create_run_dir, run_name, weights_kind


def test_run_name_is_readable():
    when = datetime(2026, 6, 12, 14, 5, 33)
    assert run_name("train", when) == "train_12-06_14h05"


def test_create_run_dir_creates_directory(tmp_path):
    path = create_run_dir(tmp_path, "eval_test")
    assert path.is_dir()
    assert path.name.startswith("eval_test_")


def test_create_run_dir_suffixes_on_collision(tmp_path):
    first = create_run_dir(tmp_path, "eval_test")
    second = create_run_dir(tmp_path, "eval_test")
    third = create_run_dir(tmp_path, "eval_test")
    assert first != second != third
    assert second.name == f"{first.name}-2"
    assert third.name == f"{first.name}-3"


def test_weights_kind_from_run_dir():
    assert weights_kind("runs/finetune_08-07_10h12/weights/best.pt") == "finetune"
    assert weights_kind("runs/pretrain_29-06_17h13/weights/last.pt") == "pretrain"
    assert weights_kind("runs/train_14-06_17h39/weights/best.pt") == "train"


def test_weights_kind_from_file_stem():
    assert weights_kind("models/pretrain_yolov8n.pt") == "pretrain"
    assert weights_kind("models/pretrain-rtdetr-l.pt") == "pretrain"
    assert weights_kind("yolov8n.pt") == "model"  # non reconnu -> générique
