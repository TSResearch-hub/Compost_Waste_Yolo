"""Tests du regroupement en sessions et du split par session."""

import pytest

from compost_detection.splitting import (
    assign_by_manifest,
    assign_split,
    build_session_mapping,
    cluster_sessions,
    load_manifest,
    parse_capture_timestamp,
    try_parse_capture_timestamp,
)

HOUR = 3600


def test_parse_valid_filename():
    assert parse_capture_timestamp("cap_1780704142") == 1780704142


@pytest.mark.parametrize("stem", ["capture_123", "cap_", "cap_12a3", "1780704142", "cap"])
def test_parse_invalid_filename_raises_clear_error(stem):
    with pytest.raises(ValueError, match="Nom de fichier invalide"):
        parse_capture_timestamp(stem)


def test_try_parse_returns_none_instead_of_raising():
    assert try_parse_capture_timestamp("cap_1780704142") == 1780704142
    assert try_parse_capture_timestamp("IMG_0042") is None


def test_close_timestamps_same_session():
    timestamps = [1780704000, 1780704060, 1780704900]  # écarts << 60 min
    mapping = cluster_sessions(timestamps, gap_minutes=60)
    assert len(set(mapping.values())) == 1


def test_gap_above_threshold_creates_distinct_sessions():
    timestamps = [1780704000, 1780704060, 1780704060 + 2 * HOUR]
    mapping = cluster_sessions(timestamps, gap_minutes=60)
    assert mapping[1780704000] == mapping[1780704060]
    assert mapping[1780704000] != mapping[1780704060 + 2 * HOUR]


def test_all_images_of_a_session_in_same_split():
    timestamps = [1780704000 + i * 30 for i in range(50)]  # plusieurs sessions
    timestamps += [1780704000 + 5 * HOUR + i * 30 for i in range(50)]
    mapping = cluster_sessions(timestamps, gap_minutes=60)
    split_by_ts = {ts: assign_split(sid) for ts, sid in mapping.items()}
    for session_id in set(mapping.values()):
        splits = {split_by_ts[ts] for ts in timestamps if mapping[ts] == session_id}
        assert len(splits) == 1


def test_adding_new_session_does_not_change_existing_assignments():
    old_timestamps = [1780704000, 1780704000 + 5 * HOUR, 1780704000 + 10 * HOUR]
    old_mapping = cluster_sessions(old_timestamps, gap_minutes=60)
    old_splits = {sid: assign_split(sid, seed=42) for sid in old_mapping.values()}

    # une nouvelle session arrive plus tard : ids et splits anciens inchangés
    new_timestamps = old_timestamps + [1780704000 + 20 * HOUR]
    new_mapping = cluster_sessions(new_timestamps, gap_minutes=60)
    for ts in old_timestamps:
        assert new_mapping[ts] == old_mapping[ts]
    for sid, split in old_splits.items():
        assert assign_split(sid, seed=42) == split


def test_assign_split_is_deterministic_and_valid():
    for sid in ("S20260605_120000", "S20260606_090000", "session_xyz"):
        split = assign_split(sid, seed=7)
        assert split in ("train", "val", "test")
        assert assign_split(sid, seed=7) == split


def test_manifest_assignment():
    sessions = [("S1", 100, 200), ("S2", 300, 400)]
    mapping = assign_by_manifest([150, 350, 400], sessions)
    assert mapping == {150: "S1", 350: "S2", 400: "S2"}


def test_manifest_rejects_orphan_capture():
    with pytest.raises(ValueError, match="aucune session"):
        assign_by_manifest([999], [("S1", 100, 200)])


def test_manifest_takes_priority_over_clustering(tmp_path):
    # timestamps proches (même session pour le clustering) mais que le
    # manifeste répartit en deux sessions distinctes
    (tmp_path / "sessions.csv").write_text(
        "session_id,start_ts,end_ts\nSA,100,150\nSB,151,300\n")
    mapping, method = build_session_mapping(tmp_path, ["cap_120", "cap_160"], gap_minutes=60)
    assert method == "manifeste sessions.csv"
    assert mapping == {"cap_120": "SA", "cap_160": "SB"}


def test_fallback_to_clustering_without_manifest(tmp_path):
    mapping, method = build_session_mapping(tmp_path, ["cap_120", "cap_160"], gap_minutes=60)
    assert method == "clustering temporel"
    assert mapping["cap_120"] == mapping["cap_160"]


def test_groups_csv_takes_priority(tmp_path):
    (tmp_path / "groups.csv").write_text("stem,group_id\nIMG_001,G1\ncap_120,G2\n")
    mapping, method = build_session_mapping(tmp_path, ["IMG_001", "cap_120"], gap_minutes=60)
    assert "groups.csv" in method
    assert mapping == {"IMG_001": "G1", "cap_120": "G2"}


def test_free_filenames_each_form_their_own_group(tmp_path):
    mapping, method = build_session_mapping(tmp_path, ["IMG_001", "IMG_002"], gap_minutes=60)
    assert mapping == {"IMG_001": "IMG_001", "IMG_002": "IMG_002"}
    assert "hors convention cap_" in method


def test_mixed_capture_and_free_filenames(tmp_path):
    stems = ["cap_120", "cap_160", "IMG_001"]
    mapping, method = build_session_mapping(tmp_path, stems, gap_minutes=60)
    assert mapping["cap_120"] == mapping["cap_160"]  # même session (clustering)
    assert mapping["IMG_001"] == "IMG_001"           # groupe singleton
    assert "clustering temporel" in method and "hors convention" in method


def test_load_manifest(tmp_path):
    path = tmp_path / "sessions.csv"
    path.write_text("session_id,start_ts,end_ts\nS1,100,200\n")
    assert load_manifest(path) == [("S1", 100, 200)]
