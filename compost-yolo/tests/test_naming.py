"""Tests du nommage des dossiers de runs."""

from datetime import datetime

from compost_detection.naming import create_run_dir, run_name


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
