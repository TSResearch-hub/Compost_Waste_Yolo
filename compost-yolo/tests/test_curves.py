"""Tests des courbes d'entraînement : lecture, recousage, sauvegarde, figures."""

import os
from pathlib import Path

import pytest

from compost_detection.curves import (column, copy_results_preserving_history,
                                      find_results_csvs, first_epoch, load_run,
                                      loss_columns, make_compare_figure,
                                      make_losses_figure, make_metrics_figure,
                                      read_results_csv, stitch)

YOLO_HEADER = ("epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,"
               "metrics/precision(B),metrics/recall(B),metrics/mAP50(B),"
               "metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss,"
               "lr/pg0,lr/pg1,lr/pg2")
RTDETR_HEADER = ("epoch,time,train/giou_loss,train/cls_loss,train/l1_loss,"
                 "metrics/precision(B),metrics/recall(B),metrics/mAP50(B),"
                 "metrics/mAP50-95(B),val/giou_loss,val/cls_loss,val/l1_loss")


def write_csv(path, header, epochs, map50=0.5):
    """Écrit un results.csv factice : epoch, time, 3 loss train, 4 métriques,
    puis les colonnes restantes (loss val, lr...) remplies uniformément."""
    lines = [header]
    extra = len(header.split(",")) - 9  # colonnes après les 4 métriques
    for e in epochs:
        lines.append(",".join([str(e), str(e * 10.0), "1.5", "4.0", "1.6",
                               "0.6", "0.4", str(map50), "0.3"] + ["1.2"] * extra))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return Path(path)


def test_read_results_csv_cleans_padded_headers(tmp_path):
    p = tmp_path / "results.csv"
    p.write_text("epoch,  train/box_loss , metrics/mAP50(B)\n1,2.0,0.5\n", encoding="utf-8")
    rows = read_results_csv(p)
    assert rows[0]["train/box_loss"] == 2.0
    assert rows[0]["metrics/mAP50(B)"] == 0.5


def test_stitch_resume_recent_segment_wins(tmp_path):
    a = read_results_csv(write_csv(tmp_path / "a.csv", YOLO_HEADER, range(1, 6), map50=0.5))
    b = read_results_csv(write_csv(tmp_path / "b.csv", YOLO_HEADER, range(4, 9), map50=0.7))
    rows = stitch([a, b])
    assert [int(r["epoch"]) for r in rows] == list(range(1, 9))
    assert column(rows, "metrics/mAP50(B)")[1][3] == 0.7  # epoch 4 : segment récent


def test_loss_columns_detects_architecture(tmp_path):
    yolo = read_results_csv(write_csv(tmp_path / "y.csv", YOLO_HEADER, [1]))
    assert loss_columns(yolo) == [("train/box_loss", "val/box_loss"),
                                  ("train/cls_loss", "val/cls_loss"),
                                  ("train/dfl_loss", "val/dfl_loss")]
    rtdetr = read_results_csv(write_csv(tmp_path / "r.csv", RTDETR_HEADER, [1]))
    assert loss_columns(rtdetr) == [("train/cls_loss", "val/cls_loss"),
                                    ("train/giou_loss", "val/giou_loss"),
                                    ("train/l1_loss", "val/l1_loss")]


def test_first_epoch(tmp_path):
    p = write_csv(tmp_path / "results.csv", YOLO_HEADER, range(4, 9))
    assert first_epoch(p) == 4.0
    assert first_epoch(tmp_path / "absent.csv") is None


def test_load_run_stitches_directory(tmp_path):
    old = write_csv(tmp_path / "results_avant_epoch_6.csv", YOLO_HEADER, range(1, 6))
    write_csv(tmp_path / "results.csv", YOLO_HEADER, range(6, 9))
    past = old.stat().st_mtime - 3600
    os.utime(old, (past, past))
    rows = load_run(tmp_path)
    assert [int(r["epoch"]) for r in rows] == list(range(1, 9))


def test_load_run_without_csv_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_run(tmp_path)


def test_copy_results_preserves_history_on_resume(tmp_path):
    save_dir, dst = tmp_path / "run", tmp_path / "backup"
    save_dir.mkdir(), dst.mkdir()
    # historique déjà sauvegardé (epochs 1-5), puis reprise qui repart à 6
    write_csv(dst / "results.csv", YOLO_HEADER, range(1, 6))
    write_csv(save_dir / "results.csv", YOLO_HEADER, range(6, 9))
    copy_results_preserving_history(save_dir, dst)
    assert first_epoch(dst / "results.csv") == 6.0
    assert first_epoch(dst / "results_avant_epoch_6.csv") == 1.0
    # backups suivants du même segment : simple écrasement, pas de nouvelle archive
    write_csv(save_dir / "results.csv", YOLO_HEADER, range(6, 12))
    copy_results_preserving_history(save_dir, dst)
    assert len(list(dst.glob("results*.csv"))) == 2


def test_figures_are_written(tmp_path):
    yolo = read_results_csv(write_csv(tmp_path / "y.csv", YOLO_HEADER, range(1, 6), map50=0.5))
    rtdetr = read_results_csv(write_csv(tmp_path / "r.csv", RTDETR_HEADER, range(1, 6), map50=0.52))
    for name, fig in [("m.png", make_metrics_figure(yolo, "yolo")),
                      ("l.png", make_losses_figure(rtdetr, "rtdetr")),
                      ("c.png", make_compare_figure({"yolo": yolo, "rtdetr": rtdetr}))]:
        out = tmp_path / name
        fig.savefig(out)
        assert out.stat().st_size > 0
