"""Worker de pré-annotation — processus séparé du serveur web.

Il ne communique avec l'application que par la base et le stockage : c'est ce
qui permettra de le déplacer sur une machine GPU sans toucher au serveur.

Décisions validées le 2026-07-31 :
- réclamation par image via `FOR UPDATE SKIP LOCKED` : deux workers ne peuvent
  jamais traiter la même image — l'accident devient inoffensif ; un crash en
  pleine inférence fait tout rollback et l'image revient dans la file, sans
  état de verrou à nettoyer. Le verrou de ligne bloque aussi tout UPDATE
  concurrent (changement de crop compris) pendant l'inférence ;
- transaction PAR IMAGE : boîtes et transition committent ensemble ;
- échec : l'image reste en `en_attente_preannotation` avec compteur, motif et
  catégorie ; au plafond de tentatives elle est « garée » (hors de la file,
  comptée à part dans GET /api/batches). Un invariant violé gare
  immédiatement ;
- reprise : les boîtes `source='modele'` ET `state='proposee'` sont écrasées
  avant réécriture — par définition jamais touchées par un humain (toute
  décision les aurait passées validee/rejetee, celles-là ne sont JAMAIS
  supprimées, pas plus que les boîtes humaines) ;
- garde systémique : N échecs consécutifs ⇒ arrêt en erreur (code 2) — des
  poids cassés sont un problème de worker, pas d'images ;
- LECTURE SEULE sur le stockage : le worker ne crée, ne modifie et ne
  supprime aucun fichier.

La transaction reste volontairement ouverte pendant l'inférence (c'est le
verrou). Sur une base hébergée, `idle_in_transaction_session_timeout` peut la
tuer en silence : le worker pose sa propre valeur de session (paramètre
USERSET) et journalise celle du serveur au démarrage.
"""
import argparse
import io
import logging
import signal
import sys
import threading
import time
from dataclasses import dataclass

from PIL import Image as PILImage
from sqlalchemy import String, cast, delete, event, func, or_, select, update

from .classes import class_count, class_names
from .config import get_settings
from .inference import Detection, InferenceEngine
from .models import Annotation, CaptureSession, Image, ImageStatusEvent
from .storage import Storage, get_storage

logger = logging.getLogger("preannotation")

STATUT_ATTENTE = "en_attente_preannotation"
# Marge large sur la plus lente inférence imaginable, mais pas l'infini :
# un worker pendu ne doit pas tenir un verrou de ligne éternellement
IDLE_TIMEOUT_WORKER = "15min"


class PanneWorker(RuntimeError):
    """Panne systémique : le worker s'arrête au lieu de brûler la file."""


class EchecImage(Exception):
    """Échec localisé à une image (kind ∈ models.PREANNOTATION_ERROR_KINDS).
    `definitif=True` gare l'image sans attendre le plafond de tentatives."""

    def __init__(self, kind: str, motif: str, *, definitif: bool = False):
        super().__init__(motif)
        self.kind = kind
        self.motif = motif
        self.definitif = definitif


@dataclass
class Stats:
    traitees: int = 0
    echecs: int = 0


class Worker:
    def __init__(self, *, sessionmaker, storage: Storage, engine: InferenceEngine,
                 max_attempts: int, poll_seconds: float,
                 max_consecutive_failures: int,
                 stop_event: threading.Event | None = None):
        self.sessionmaker = sessionmaker
        self.storage = storage
        self.engine = engine
        self.max_attempts = max_attempts
        self.poll_seconds = poll_seconds
        self.max_consecutive_failures = max_consecutive_failures
        self.stop_event = stop_event or threading.Event()
        self.stats = Stats()

    def run(self, *, drain: bool = False) -> None:
        """Boucle principale. `drain=True` : s'arrête quand la file est vide
        (tests, exécution ponctuelle) au lieu de dormir puis réessayer."""
        consecutifs = 0
        while not self.stop_event.is_set():
            resultat = self._process_one()
            if resultat is None:  # file vide
                if drain:
                    return
                self.stop_event.wait(self.poll_seconds)
                continue
            if resultat:
                consecutifs = 0
            else:
                consecutifs += 1
                if consecutifs >= self.max_consecutive_failures:
                    raise PanneWorker(
                        f"{consecutifs} échecs consécutifs — problème de "
                        "worker (poids ? stockage ?), pas d'images : arrêt"
                    )

    def _process_one(self) -> bool | None:
        """Une image, une transaction. Renvoie None si la file est vide,
        True si l'image est passée en `a_annoter`, False sur échec enregistré."""
        db = self.sessionmaker()
        image_id = None
        try:
            image = db.scalars(
                select(Image)
                .where(Image.status == STATUT_ATTENTE,
                       Image.superseded_at.is_(None),
                       Image.preannotation_attempts < self.max_attempts)
                .order_by(Image.id)
                .limit(1)
                .with_for_update(skip_locked=True)
            ).first()
            if image is None:
                db.rollback()
                return None
            image_id = image.id

            debut = time.monotonic()
            detections = self._inferer(db, image)
            db.execute(delete(Annotation).where(
                Annotation.image_id == image.id,
                Annotation.source == "modele",
                Annotation.state == "proposee"))
            for d in detections:
                db.add(Annotation(
                    image_id=image.id, class_id=d.class_id,
                    x_center=d.x_center, y_center=d.y_center,
                    box_width=d.box_width, box_height=d.box_height,
                    source="modele", state="proposee",
                    confidence=d.confidence, model_name=self.engine.model_name))
            image.status = "a_annoter"
            image.preannotation_error = None
            image.preannotation_error_kind = None
            db.add(ImageStatusEvent(
                image_id=image.id, from_status=STATUT_ATTENTE,
                to_status="a_annoter", changed_by=None))  # NULL : le système
            db.commit()
            self.stats.traitees += 1
            logger.info("image %d : %d boîte(s) proposée(s) en %.0f ms",
                        image_id, len(detections),
                        (time.monotonic() - debut) * 1000)
            return True
        except EchecImage as echec:
            db.rollback()  # libère le verrou, ne laisse rien de partiel
            self._enregistrer_echec(db, image_id, echec)
            self.stats.echecs += 1
            return False
        finally:
            db.close()

    def _inferer(self, db, image: Image) -> list[Detection]:
        etrangere = db.scalar(
            select(Annotation.id)
            .where(Annotation.image_id == image.id,
                   or_(Annotation.source != "modele",
                       Annotation.state != "proposee"))
            .limit(1))
        if etrangere is not None:
            raise EchecImage(
                "invariant_viole",
                "annotation humaine ou décidée sur une image en attente de "
                "pré-annotation — à inspecter manuellement, rien n'a été modifié",
                definitif=True)

        # Le fichier annoté : le crop s'il existe, l'original sinon —
        # les coordonnées produites lui sont relatives
        rel = image.cropped_path or image.original_path
        try:
            pil = PILImage.open(io.BytesIO(self.storage.read(rel)))
            pil.load()
        except Exception as exc:
            raise EchecImage("fichier_illisible", f"{rel} : {exc}")

        try:
            detections = self.engine.infer(pil)
        except Exception as exc:
            raise EchecImage("moteur_indisponible", str(exc))

        nc = class_count()
        for d in detections:
            if not 0 <= d.class_id < nc:
                raise EchecImage(
                    "moteur_indisponible",
                    f"class_id {d.class_id} hors référentiel (nc={nc}) — "
                    "poids incompatibles avec data.yaml ?")
        return detections

    def _enregistrer_echec(self, db, image_id: int, echec: EchecImage) -> None:
        """Petite transaction séparée, gardée sur le statut : entre le rollback
        et cette écriture, un worker rival a pu traiter l'image avec succès —
        dans ce cas on n'enregistre rien."""
        valeurs = {
            "preannotation_error": echec.motif,
            "preannotation_error_kind": echec.kind,
            "preannotation_attempts": (
                self.max_attempts if echec.definitif
                else Image.preannotation_attempts + 1
            ),
        }
        tentatives = db.scalar(
            update(Image)
            .where(Image.id == image_id, Image.status == STATUT_ATTENTE)
            .values(**valeurs)
            .returning(Image.preannotation_attempts))
        db.commit()
        if tentatives is None:
            logger.warning("image %d : échec (%s) mais l'image a changé de "
                           "statut entre-temps — rien enregistré",
                           image_id, echec.kind)
        elif tentatives >= self.max_attempts:
            logger.error("image %d GARÉE (%s) après %d tentative(s) : %s",
                         image_id, echec.kind, tentatives, echec.motif)
        else:
            logger.warning("image %d : échec (%s), tentative %d/%d : %s",
                           image_id, echec.kind, tentatives,
                           self.max_attempts, echec.motif)


# ═══ Commande d'essai (calibration du seuil) ═════════════════════════════════


def _stats_comptes(comptes: list[int]) -> str:
    n = len(comptes)
    tri = sorted(comptes)
    return (f"médiane {tri[n // 2]}, p90 {tri[min(int(n * 0.9), n - 1)]}, "
            f"max {tri[-1]}, moyenne {sum(tri) / n:.1f}")


def essai(db, storage: Storage, engine: InferenceEngine, *,
          echantillon: int, seuils: list[float]) -> str:
    """Calibration en LECTURE SEULE : aucune écriture, ni base ni stockage,
    aucun statut ni compteur modifié.

    Échantillon STRATIFIÉ : tiré à parts égales sur chaque session — les
    conditions de capture diffèrent, le seuil doit tenir sur toutes — dans un
    ordre pseudo-aléatoire déterministe (md5 de l'id) qui mêle les postes et
    rend la mesure reproductible. Les images déjà annotées (vérité terrain en
    base) sont préférées à la file d'attente : elles séparent images
    PORTEUSES de boîtes et NÉGATIFS, or sur un négatif toute détection est
    fausse — le pire cas pour l'annotateur, qui doit tout supprimer sans rien
    garder. À défaut d'images annotées, repli sur la file, sans distinction.

    Le moteur doit être configuré au seuil MINIMAL demandé et avec un plafond
    de détections large : une seule inférence par image, les comptes par seuil
    sont dérivés des confiances. On calibre sur des chiffres, pas sur un
    principe."""
    def tirer(statuts: tuple[str, ...]):
        porteuse = (
            select(Annotation.id)
            .where(Annotation.image_id == Image.id,
                   Annotation.state == "validee")
            .exists().label("porteuse")
        )
        rang = func.row_number().over(
            partition_by=Image.session_id,
            order_by=func.md5(cast(Image.id, String))).label("rang")
        interne = (
            select(Image.cropped_path, Image.original_path,
                   CaptureSession.name.label("session"), porteuse, rang)
            .join(CaptureSession, CaptureSession.id == Image.session_id)
            .where(Image.status.in_(statuts), Image.superseded_at.is_(None))
            .subquery()
        )
        # coupe à l'effectif demandé en préservant l'équilibre : un rang
        # complet de sessions entre avant le rang suivant
        return db.execute(
            select(interne)
            .order_by(interne.c.rang, interne.c.session)
            .limit(echantillon)
        ).all()

    lignes_db = tirer(("annotee", "relue"))
    verite_connue = bool(lignes_db)
    if not lignes_db:
        lignes_db = tirer((STATUT_ATTENTE,))
    if not lignes_db:
        return "Aucune image à mesurer (ni annotée, ni en attente)."

    donnees = []  # (session, porteuse, confiances)
    for crop, orig, session, porteuse, _rang in lignes_db:
        pil = PILImage.open(io.BytesIO(storage.read(crop or orig)))
        pil.load()
        donnees.append(
            (session, porteuse, [d.confidence for d in engine.infer(pil)]))

    n = len(donnees)
    par_session: dict[str, int] = {}
    for session, _, _ in donnees:
        par_session[session] = par_session.get(session, 0) + 1
    compo = ", ".join(f"{s} : {c}" for s, c in sorted(par_session.items()))
    lignes = [f"Essai sur {n} image(s) ({compo}), modèle {engine.model_name} "
              "— boîtes par image selon le seuil :"]
    if verite_connue:
        nb_port = sum(1 for _, p, _ in donnees if p)
        lignes.append(f"  vérité terrain : {nb_port} porteuse(s) de boîtes, "
                      f"{n - nb_port} négatif(s) — compost nu")
    else:
        lignes.append("  file d'attente : vérité terrain inconnue, pas de "
                      "distinction porteuses/négatifs")
    for seuil in sorted(seuils):
        tous = [sum(1 for c in confs if c >= seuil)
                for _, _, confs in donnees]
        lignes.append(f"  seuil {seuil:.2f} : {_stats_comptes(tous)}")
        if verite_connue:
            port = [sum(1 for c in confs if c >= seuil)
                    for _, p, confs in donnees if p]
            neg = [sum(1 for c in confs if c >= seuil)
                   for _, p, confs in donnees if not p]
            if port:
                lignes.append(f"    porteuses ({len(port)}) : "
                              f"{_stats_comptes(port)}")
            if neg:
                propres = sum(1 for x in neg if x == 0)
                lignes.append(f"    négatifs  ({len(neg)}) : "
                              f"{_stats_comptes(neg)} — toute boîte est "
                              f"fausse ; {propres}/{len(neg)} sans aucune")
    lignes.append("Aucune écriture effectuée.")
    return "\n".join(lignes)


# ═══ Point d'entrée ══════════════════════════════════════════════════════════


def _proteger_transactions(engine_sql) -> None:
    """Pose le timeout de session du worker sur chaque connexion : la base
    hébergée ne doit pas tuer une transaction pendant l'inférence."""
    @event.listens_for(engine_sql, "connect")
    def _set_timeout(dbapi_conn, _record):
        with dbapi_conn.cursor() as cur:
            cur.execute(
                f"SET idle_in_transaction_session_timeout = '{IDLE_TIMEOUT_WORKER}'"
            )


def _journaliser_timeout(engine_sql) -> None:
    with engine_sql.connect() as conn:
        effectif = conn.exec_driver_sql(
            "SHOW idle_in_transaction_session_timeout").scalar()
        serveur = conn.exec_driver_sql(
            "SELECT reset_val FROM pg_settings"
            " WHERE name = 'idle_in_transaction_session_timeout'").scalar()
    logger.info("idle_in_transaction_session_timeout : %s pour ce worker "
                "(valeur serveur : %s)", effectif, serveur)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m app.worker",
        description="Worker de pré-annotation (lit la base et le stockage, "
                    "n'écrit jamais dans le stockage)")
    sub = parser.add_subparsers(dest="commande")
    p_run = sub.add_parser("run", help="boucle de pré-annotation (défaut)")
    p_run.add_argument("--une-passe", action="store_true",
                       help="s'arrête quand la file est vide au lieu d'attendre")
    p_essai = sub.add_parser(
        "essai",
        help="calibration : infère un échantillon SANS RIEN ÉCRIRE et "
             "rapporte la distribution du nombre de boîtes par seuil")
    p_essai.add_argument("--echantillon", type=int, default=30)
    p_essai.add_argument("--seuils", default="0.05,0.10,0.25,0.40",
                         help="seuils de confiance à comparer (défaut : "
                              "0.05,0.10,0.25,0.40)")
    p_essai.add_argument("--plafond-mesure", type=int, default=300,
                         help="max_det pendant la mesure : large exprès, pour "
                              "voir ce que le plafond de production couperait")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    settings = get_settings()
    if settings.weights_path is None:
        raise SystemExit("WEIGHTS_PATH manquant : le worker ne démarre pas sans poids")

    from .db import get_engine, get_sessionmaker
    from .inference import UltralyticsEngine

    _proteger_transactions(get_engine())
    _journaliser_timeout(get_engine())
    storage = get_storage()

    if args.commande == "essai":
        seuils = [float(s) for s in args.seuils.split(",")]
        moteur = UltralyticsEngine(
            settings.weights_path, conf=min(seuils),
            max_det=args.plafond_mesure, expected_names=class_names())
        db = get_sessionmaker()()
        try:
            print(essai(db, storage, moteur,
                        echantillon=args.echantillon, seuils=seuils))
        finally:
            db.close()
        return

    moteur = UltralyticsEngine(
        settings.weights_path, conf=settings.worker_conf_threshold,
        max_det=settings.worker_max_det, expected_names=class_names())
    worker = Worker(
        sessionmaker=get_sessionmaker(), storage=storage, engine=moteur,
        max_attempts=settings.worker_max_attempts,
        poll_seconds=settings.worker_poll_seconds,
        max_consecutive_failures=settings.worker_max_consecutive_failures)

    def _arret(_sig, _frame):
        logger.info("arrêt demandé — l'image en cours se termine, "
                    "le worker ne repart pas sur la suivante")
        worker.stop_event.set()

    signal.signal(signal.SIGINT, _arret)
    signal.signal(signal.SIGTERM, _arret)

    try:
        worker.run(drain=getattr(args, "une_passe", False))
    except PanneWorker as exc:
        logger.error("%s", exc)
        logger.info("bilan : %d traitée(s), %d échec(s)",
                    worker.stats.traitees, worker.stats.echecs)
        sys.exit(2)
    logger.info("bilan : %d traitée(s), %d échec(s)",
                worker.stats.traitees, worker.stats.echecs)


if __name__ == "__main__":
    main()
