"""Reconstruction des groupes de capture et split train/val/test par groupe.

Le split se fait au niveau de GROUPES d'images (jamais image par image, sauf
choix explicite) pour éviter la fuite de données entre train et val/test.
Trois façons de grouper, par ordre de priorité :

1. ``groups.csv`` dans le dossier source (colonnes : stem,group_id) — généré
   par scripts/import_dataset.py pour les datasets externes ;
2. pour les noms ``cap_{timestamp_unix}`` de l'interface d'annotation :
   manifeste ``sessions.csv`` si présent, sinon clustering temporel ;
3. les fichiers hors convention ``cap_`` et absents de groups.csv forment
   chacun leur propre groupe (équivaut à un split aléatoire par image).
"""

import csv
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

CAPTURE_PATTERN = re.compile(r"^cap_(\d+)$")


def try_parse_capture_timestamp(stem):
    """Timestamp Unix d'un nom ``cap_{timestamp_unix}``, ou None sinon."""
    match = CAPTURE_PATTERN.match(stem)
    return int(match.group(1)) if match else None


def parse_capture_timestamp(stem):
    """Extrait le timestamp Unix d'un nom de capture (sans extension).

    Lève ValueError avec un message clair si le nom ne respecte pas la
    convention ``cap_{timestamp_unix}``.
    """
    ts = try_parse_capture_timestamp(stem)
    if ts is None:
        raise ValueError(
            f"Nom de fichier invalide : '{stem}' "
            "(attendu : cap_<timestamp_unix>, ex. cap_1780704142)"
        )
    return ts


def load_manifest(csv_path):
    """Lit un manifeste sessions.csv (colonnes : session_id,start_ts,end_ts)."""
    sessions = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sessions.append((row["session_id"], int(row["start_ts"]), int(row["end_ts"])))
    return sessions


def assign_by_manifest(timestamps, sessions):
    """Affecte chaque timestamp à sa session du manifeste.

    Retourne {timestamp: session_id}. Lève ValueError si une capture ne
    tombe dans aucune session déclarée.
    """
    mapping = {}
    for ts in timestamps:
        for session_id, start_ts, end_ts in sessions:
            if start_ts <= ts <= end_ts:
                mapping[ts] = session_id
                break
        else:
            raise ValueError(
                f"La capture cap_{ts} ne correspond à aucune session du manifeste "
                "(vérifier les colonnes start_ts/end_ts de sessions.csv)"
            )
    return mapping


def cluster_sessions(timestamps, gap_minutes=60):
    """Regroupe les timestamps en sessions par clustering temporel.

    Deux captures séparées de plus de ``gap_minutes`` appartiennent à des
    sessions différentes. Retourne {timestamp: session_id}.

    L'id de session est dérivé UNIQUEMENT du premier timestamp du cluster
    (``S{date}_{heure}``) : ajouter de nouvelles captures plus tard ne change
    jamais l'id (donc le split) des sessions existantes.
    """
    gap_seconds = gap_minutes * 60
    mapping = {}
    session_id = None
    prev_ts = None
    for ts in sorted(timestamps):
        if prev_ts is None or ts - prev_ts > gap_seconds:
            first = datetime.fromtimestamp(ts, tz=timezone.utc)
            session_id = f"S{first:%Y%m%d_%H%M%S}"
        mapping[ts] = session_id
        prev_ts = ts
    return mapping


def load_groups(csv_path):
    """Lit un groups.csv (colonnes : stem,group_id). Retourne {stem: group_id}."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {row["stem"]: row["group_id"] for row in csv.DictReader(f)}


def build_session_mapping(source_dir, stems, gap_minutes=60):
    """Affecte chaque nom de fichier (sans extension) à un groupe de split.

    Priorités par fichier : groups.csv s'il l'y déclare ; sinon, pour les noms
    ``cap_{ts}``, manifeste sessions.csv puis clustering temporel ; sinon le
    fichier forme son propre groupe (split par image).

    Retourne (mapping {stem: group_id}, méthode : description lisible des
    regroupements utilisés, pour affichage).
    """
    source_dir = Path(source_dir)
    groups_path = source_dir / "groups.csv"
    groups = load_groups(groups_path) if groups_path.exists() else {}
    mapping = {}
    timed = {}  # stems suivant la convention cap_{ts}
    singletons = 0
    for stem in stems:
        if stem in groups:
            mapping[stem] = groups[stem]
        elif (ts := try_parse_capture_timestamp(stem)) is not None:
            timed[stem] = ts
        else:
            mapping[stem] = stem
            singletons += 1

    methods = []
    if groups:
        methods.append("groups.csv")
    if timed:
        manifest_path = source_dir / "sessions.csv"
        if manifest_path.exists():
            by_ts = assign_by_manifest(timed.values(), load_manifest(manifest_path))
            methods.append("manifeste sessions.csv")
        else:
            by_ts = cluster_sessions(timed.values(), gap_minutes)
            methods.append("clustering temporel")
        for stem, ts in timed.items():
            mapping[stem] = by_ts[ts]
    if singletons:
        methods.append(f"{singletons} fichiers hors convention cap_ -> 1 groupe par image")
    return mapping, " + ".join(methods) if methods else "aucun fichier"


def assign_split(session_id, ratios=(0.70, 0.15, 0.15), seed=0):
    """Affecte une session à 'train', 'val' ou 'test' de façon déterministe.

    Le choix dépend uniquement du hash de (seed, session_id) : ajouter ou
    retirer d'autres sessions ne change JAMAIS l'affectation d'une session
    existante (comparabilité des métriques, pas de contamination train→test).
    """
    digest = hashlib.md5(f"{seed}:{session_id}".encode()).hexdigest()
    x = int(digest[:8], 16) / 0xFFFFFFFF
    if x < ratios[0]:
        return "train"
    if x < ratios[0] + ratios[1]:
        return "val"
    return "test"
