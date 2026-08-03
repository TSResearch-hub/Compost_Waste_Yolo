"""TEST D'ACCEPTATION : l'aller-retour sur le dataset réel.

Couverture (les motifs partitionnent images/), puis les SIX passes de reprise
explicite — deux sessions de capture, six postes —, export YOLO, et
comparaison NUMÉRIQUE à la source avec tolérance 1e-6 : les coordonnées
transitent par un double, leur écriture décimale n'est pas octet pour octet
celle d'origine. Un écart = la reprise ou l'export perd de l'information, et
il faut le savoir AVANT de peupler la base réelle.

Convention (2026-08-03) : l'absence de fichier label = NÉGATIF (compost nu),
pas annotation manquante — l'image entre en `annotee` sans boîte et ressort
avec un .txt vide de 0 octet.
"""
import fnmatch
from datetime import date
from pathlib import Path

import pytest

from conftest import exec_sql

REPO = Path(__file__).resolve().parents[2]
IMAGES_SRC = REPO / "dataset_recolte" / "images"
LABELS_SRC = REPO / "dataset_recolte" / "labels"

TOLERANCE = 1e-6

# ── Le contenu réel du dataset aplati : 2 sessions, 6 postes (décision Réda
# du 2026-08-03). Une création par session (date fournie), puis rattachements.
PASSES = [
    ("session_2026-06-18", date(2026, 6, 18), "webcam_avant_pause",
     "captures_1_images_WIN_20260618_*"),
    ("session_2026-06-18", None, "webcam_apres_pause",
     "captures_1.5_images_WIN_20260618_*"),
    ("session_2026-06-18", None, "telephone_hd", "20260618_*"),
    ("session_2026-07-14", date(2026, 7, 14), "ali", "20260714_*_ali.jpg"),
    ("session_2026-07-14", None, "hamza", "20260714_*_hamza.jpg"),
    ("session_2026-07-14", None, "reda", "IMG_*_reda.jpg"),
]


def parse_label(path: Path):
    """Boîtes triées d'un fichier YOLO (tolère l'absence de \\n final,
    ignore les lignes vides)."""
    boites = []
    for raw in path.read_text().splitlines():
        parts = raw.split()
        if parts:
            boites.append((int(parts[0]), *(float(v) for v in parts[1:])))
    return sorted(boites)


@pytest.mark.skipif(not (IMAGES_SRC.is_dir() and LABELS_SRC.is_dir()),
                    reason="dataset_recolte absent de cette machine")
def test_aller_retour_dataset_historique(db, engine, make_user, tmp_path):
    from app.exporter import export_yolo
    from app.importer import (import_historical_session,
                              plan_historical_session, verifier_couverture)
    from app.storage import get_storage

    admin = make_user("root", role="administrateur")
    fichiers = {p.stem: p for p in IMAGES_SRC.iterdir() if p.is_file()}

    # 0. Couverture : les six motifs partitionnent images/, aucun label orphelin
    couverture = verifier_couverture(
        images_dir=IMAGES_SRC, labels_dir=LABELS_SRC,
        patterns=[motif for *_, motif in PASSES])
    assert couverture.ok, couverture.summary()

    # 1. Les six passes de reprise, dans l'ordre de la table
    session_ids: dict[str, int] = {}
    stats = []  # (session, poste, images, labels, négatifs, boîtes)
    for session_nom, jour, poste, motif in PASSES:
        plan = plan_historical_session(
            db, images_dir=IMAGES_SRC, labels_dir=LABELS_SRC, pattern=motif,
            session_name=session_nom, source_label=poste, captured_on=jour)
        assert plan.rejected == [] and plan.duplicates == [], plan.summary()
        # la convention tient : seuls les fichiers SANS .txt sont sans boîte
        # (un .txt vide ferait diverger les deux comptes — à signaler)
        porteurs = sum(1 for p in IMAGES_SRC.glob(motif)
                       if (LABELS_SRC / f"{p.stem}.txt").is_file())
        assert plan.negatives == plan.total_images - porteurs
        rapport = import_historical_session(db, get_storage(), plan=plan,
                                            admin_id=admin.id)
        assert rapport.renamed == []  # noms tous distincts dans la session
        session_ids[session_nom] = rapport.session_id
        stats.append((session_nom, poste, plan.total_images, porteurs,
                      plan.negatives, plan.total_boxes))

    lignes_tbl = [f"{'session':22s} {'poste':20s} {'images':>6s} "
                  f"{'labels':>6s} {'négatifs':>8s} {'boîtes':>6s}"]
    for session_nom, poste, ni, nl, nn, nb in stats:
        lignes_tbl.append(f"{session_nom:22s} {poste:20s} {ni:6d} "
                          f"{nl:6d} {nn:8d} {nb:6d}")
    tot = [sum(s[i] for s in stats) for i in (2, 3, 4, 5)]
    lignes_tbl.append(f"{'TOTAL':43s} {tot[0]:6d} {tot[1]:6d} "
                      f"{tot[2]:8d} {tot[3]:6d}")
    print("\nDécompte par session et par poste :\n" + "\n".join(lignes_tbl))

    # tout images/ est en base, sous six postes distincts et deux sessions
    assert tot[0] == len(fichiers)
    assert exec_sql(engine, "SELECT count(*) FROM images") == len(fichiers)
    assert exec_sql(engine,
                    "SELECT count(DISTINCT source_label) FROM images") == 6
    assert exec_sql(engine, "SELECT count(*) FROM sessions") == 2

    # 2. Export
    sortie = tmp_path / "aller_retour"
    report = export_yolo(db, get_storage(), output_dir=sortie)
    print("\n" + report.summary())

    # 3. Comparaison à la source : CHAQUE image de images/ ressort — celles
    # sans fichier label en .txt vide de 0 octet
    exportes = {p.stem: p for p in (sortie / "labels").glob("*.txt")}
    assert set(exportes) == set(fichiers), (
        f"manquants : {sorted(set(fichiers) - set(exportes))[:5]} / "
        f"en trop : {sorted(set(exportes) - set(fichiers))[:5]}")
    assert len(list((sortie / "images").iterdir())) == len(fichiers)

    ecarts = []
    negatifs_source = 0
    for stem in fichiers:
        src_path = LABELS_SRC / f"{stem}.txt"
        src = parse_label(src_path) if src_path.is_file() else []
        exp = parse_label(exportes[stem])
        if not src:
            negatifs_source += 1
        if len(src) != len(exp):
            ecarts.append(f"{stem} : {len(src)} boîte(s) source, "
                          f"{len(exp)} exportée(s)")
            continue
        for a, b in zip(src, exp):
            if a[0] != b[0]:  # index de classe
                ecarts.append(f"{stem} : classe {a[0]} devenue {b[0]}")
            elif any(abs(x - y) > TOLERANCE for x, y in zip(a[1:], b[1:])):
                ecarts.append(f"{stem} : coordonnées au-delà de {TOLERANCE}")
    assert ecarts == [], "l'aller-retour perd de l'information :\n" + \
        "\n".join(ecarts[:10])
    # les négatifs ressortent en .txt de 0 octet exactement
    assert all(exportes[s].stat().st_size == 0
               for s in fichiers if not (LABELS_SRC / f"{s}.txt").is_file())

    # groups.csv : 2 groupes ; chaque image sous l'IDENTIFIANT de sa session,
    # déterminé par son motif — le split par session est reconstituable
    lignes = (sortie / "groups.csv").read_text().splitlines()
    assert lignes[0] == "stem,group_id"
    groupe_de = dict(ligne.rsplit(",", 1) for ligne in lignes[1:])
    assert len(groupe_de) == len(fichiers)
    assert set(groupe_de.values()) == {str(i) for i in session_ids.values()}
    attendu = {}
    for session_nom, _, _, motif in PASSES:
        for p in IMAGES_SRC.glob(motif):
            attendu[p.stem] = str(session_ids[session_nom])
    assert groupe_de == attendu

    # cohérence du rapport d'export avec la source
    assert report.images == len(fichiers)
    assert report.boxes == sum(len(parse_label(p))
                               for p in LABELS_SRC.glob("*.txt"))
    assert report.empty_labels == negatifs_source == tot[2]
