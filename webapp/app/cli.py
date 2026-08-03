"""CLI d'administration.

    python -m app.cli create-admin --username reda [--display-name "Réda"]
    python -m app.cli import-session --source /depot/2026-07-18 \
        --name 2026-07-18_tas-A --date 2026-07-18 --admin reda [options]

create-admin : mot de passe demandé interactivement (deux saisies), ou lu
depuis une variable d'environnement avec --password-env NOM (scripts/tests).
Ne crée que le PREMIER administrateur : s'il en existe déjà un actif, refus —
les comptes suivants se créent via l'API, connecté en administrateur.

import-session : importe un dossier d'images déposé côté serveur (lecture
seule sur la source) en une session + lot « import ». Rapport en fin de
commande ; code de sortie 1 si l'import est refusé (rien d'importable).

couverture : lecture seule, à lancer AVANT toute reprise. Vérifie que les
motifs fournis (--motif répétable) partitionnent exactement le dossier
d'images : tout fichier hors de tout motif, ou sous plusieurs motifs, est
listé et la commande échoue (code 1) — l'oubli silencieux devient un échec
bruyant. Signale aussi les labels sans image.

import-historique : reprise d'un dataset déjà annoté (images + labels YOLO)
vers une session EXPLICITE, un poste de capture par commande : --session
désigne la cible (avec --date : création ; sans --date : rattachement à une
session existante), --poste le poste de capture (source_label), --motif les
fichiers concernés. L'absence de fichier label = NÉGATIF (compost nu) :
l'image entre en statut `annotee` avec zéro boîte. Les fichiers sources ne
sont jamais renommés ; annotations attribuées au compte inactif
`import_historique`. Par défaut la commande ANALYSE SEULEMENT (rien n'est
écrit) ; l'écriture exige --execute. Ne jamais lancer --execute sans ordre
explicite.

export-yolo : exporte le dataset YOLO (images/, labels/, groups.csv,
classes.txt) vers un répertoire vide ou inexistant — consommable tel quel par
compost-yolo/scripts/prepare_dataset.py. Sans --session : tout.
"""
import argparse
import getpass
import os
import sys
from datetime import date
from pathlib import Path

from sqlalchemy import select

from .db import get_sessionmaker
from .models import User
from .security import PASSWORD_MIN_LENGTH, hash_password


def _read_password(password_env: str | None) -> str:
    if password_env:
        password = os.environ.get(password_env)
        if not password:
            sys.exit(f"Variable d'environnement {password_env} vide ou absente.")
        return password
    password = getpass.getpass("Mot de passe : ")
    if password != getpass.getpass("Confirmation : "):
        sys.exit("Les deux saisies ne correspondent pas.")
    return password


def create_admin(args: argparse.Namespace) -> None:
    password = _read_password(args.password_env)
    if len(password) < PASSWORD_MIN_LENGTH:
        sys.exit(f"Mot de passe trop court (minimum {PASSWORD_MIN_LENGTH} caractères).")

    with get_sessionmaker()() as db:
        existing = db.scalar(
            select(User).where(User.role == "administrateur", User.is_active)
        )
        if existing is not None:
            sys.exit(
                f"Un administrateur actif existe déjà ({existing.username}) : "
                "les comptes suivants se créent via l'API."
            )
        user = User(
            username=args.username,
            password_hash=hash_password(password),
            display_name=args.display_name,
            role="administrateur",
        )
        db.add(user)
        db.commit()
        print(f"Administrateur créé : {user.username} (id {user.id})")


def import_session(args: argparse.Namespace) -> None:
    from .importer import import_session_folder
    from .storage import get_storage

    try:
        captured_on = date.fromisoformat(args.date)
    except ValueError:
        sys.exit(f"Date invalide : {args.date} (format attendu AAAA-MM-JJ).")

    with get_sessionmaker()() as db:
        admin = db.scalar(
            select(User).where(
                User.username == args.admin,
                User.role == "administrateur",
                User.is_active,
            )
        )
        if admin is None:
            sys.exit(f"« {args.admin} » n'est pas un administrateur actif.")
        try:
            report = import_session_folder(
                db, get_storage(),
                source_dirs=[Path(s) for s in args.source], admin_id=admin.id,
                name=args.name, captured_on=captured_on,
                lighting=args.lighting, camera_height_cm=args.camera_height_cm,
                compost_state=args.compost_state, operator=args.operator,
                notes=args.notes,
            )
        except ValueError as exc:
            sys.exit(str(exc))
    print(report.summary())
    if report.aborted:
        sys.exit(1)


def import_historique(args: argparse.Namespace) -> None:
    from .importer import import_historical_session, plan_historical_session
    from .storage import get_storage

    captured_on = None
    if args.date:
        try:
            captured_on = date.fromisoformat(args.date)
        except ValueError:
            sys.exit(f"Date invalide : {args.date} (format attendu AAAA-MM-JJ).")

    with get_sessionmaker()() as db:
        admin = db.scalar(
            select(User).where(
                User.username == args.admin,
                User.role == "administrateur",
                User.is_active,
            )
        )
        if admin is None:
            sys.exit(f"« {args.admin} » n'est pas un administrateur actif.")
        try:
            plan = plan_historical_session(
                db, images_dir=Path(args.images),
                labels_dir=Path(args.labels), pattern=args.motif,
                session_name=args.session, source_label=args.poste,
                captured_on=captured_on,
            )
        except ValueError as exc:
            sys.exit(str(exc))
        print(plan.summary())
        if not args.execute:
            print("\nANALYSE SEULE — rien n'a été écrit. "
                  "Relancer avec --execute pour importer.")
            return
        if plan.total_images == 0:
            print("\nRien à importer.")
            sys.exit(1)
        try:
            report = import_historical_session(
                db, get_storage(), plan=plan, admin_id=admin.id)
        except ValueError as exc:
            sys.exit(str(exc))
        print(f"\nImport exécuté : {len(report.created)} image(s), dont "
              f"{plan.negatives} négatif(s), → session "
              f"« {report.session_name} » (id {report.session_id})")
        for brut, export in report.renamed:
            print(f"  renommé pour l'export : {brut} → {export}")


def couverture(args: argparse.Namespace) -> None:
    from .importer import verifier_couverture

    try:
        report = verifier_couverture(
            images_dir=Path(args.images), labels_dir=Path(args.labels),
            patterns=args.motif,
        )
    except ValueError as exc:
        sys.exit(str(exc))
    print(report.summary())
    if not report.ok:
        sys.exit(1)


def export_yolo_cmd(args: argparse.Namespace) -> None:
    from .exporter import export_yolo
    from .storage import get_storage

    with get_sessionmaker()() as db:
        try:
            report = export_yolo(db, get_storage(),
                                 output_dir=Path(args.output),
                                 session_names=args.session)
        except ValueError as exc:
            sys.exit(str(exc))
    print(report.summary())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m app.cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_admin = sub.add_parser("create-admin", help="créer le premier administrateur")
    p_admin.add_argument("--username", required=True)
    p_admin.add_argument("--display-name", default=None)
    p_admin.add_argument(
        "--password-env",
        default=None,
        metavar="VAR",
        help="lire le mot de passe depuis cette variable d'environnement",
    )
    p_admin.set_defaults(func=create_admin)

    p_import = sub.add_parser(
        "import-session",
        help="importer un dossier d'images (lecture seule sur la source)",
    )
    p_import.add_argument("--source", required=True, action="append",
                          help="dossier serveur d'images ; répétable — un par "
                               "poste de capture, la session reste UNIQUE")
    p_import.add_argument("--name", required=True, help="nom unique de la session")
    p_import.add_argument("--date", required=True,
                          help="date de capture (AAAA-MM-JJ)")
    p_import.add_argument("--admin", required=True,
                          help="username de l'administrateur qui importe")
    p_import.add_argument("--lighting", default=None, help="éclairage")
    p_import.add_argument("--camera-height-cm", type=int, default=None)
    p_import.add_argument("--compost-state", default=None, help="état du compost")
    p_import.add_argument("--operator", default=None, help="opérateur terrain")
    p_import.add_argument("--notes", default=None)
    p_import.set_defaults(func=import_session)

    p_couv = sub.add_parser(
        "couverture",
        help="vérifier que les motifs partitionnent le dossier d'images "
             "(lecture seule, à lancer avant toute reprise)",
    )
    p_couv.add_argument("--images", required=True, help="dossier des images")
    p_couv.add_argument("--labels", required=True, help="dossier des labels YOLO")
    p_couv.add_argument("--motif", required=True, action="append",
                        help="motif glob ; répétable — un par poste de capture")
    p_couv.set_defaults(func=couverture)

    p_hist = sub.add_parser(
        "import-historique",
        help="reprise d'un dataset annoté vers une session explicite "
             "(analyse seule sans --execute)",
    )
    p_hist.add_argument("--images", required=True, help="dossier des images")
    p_hist.add_argument("--labels", required=True, help="dossier des labels YOLO")
    p_hist.add_argument("--admin", required=True,
                        help="username de l'administrateur qui importe")
    p_hist.add_argument("--execute", action="store_true",
                        help="écrire réellement (sinon : analyse seule)")
    p_hist.add_argument("--session", required=True,
                        help="nom de la session cible (avec --date : création ; "
                             "sans --date : rattachement à une session existante)")
    p_hist.add_argument("--date", default=None,
                        help="date de capture (AAAA-MM-JJ) si la session doit "
                             "être créée ; omise = rattachement")
    p_hist.add_argument("--poste", required=True,
                        help="libellé du poste de capture (source_label) des "
                             "images reprises — ex. telephone_hd")
    p_hist.add_argument("--motif", required=True,
                        help='motif glob des fichiers concernés, ex. "IMG_*" '
                             "— les autres fichiers relèvent d'autres passes")
    p_hist.set_defaults(func=import_historique)

    p_export = sub.add_parser(
        "export-yolo",
        help="exporter le dataset YOLO vers un répertoire vide ou inexistant",
    )
    p_export.add_argument("--output", required=True,
                          help="répertoire de sortie (vide ou inexistant)")
    p_export.add_argument("--session", action="append", default=None,
                          help="nom de session à exporter ; répétable — "
                               "sans cette option : tout")
    p_export.set_defaults(func=export_yolo_cmd)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
