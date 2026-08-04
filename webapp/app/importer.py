"""Import d'un lot d'images depuis un dossier déposé côté serveur.

Règles de périmètre (fixées le 2026-07-28) :
- le dossier source est en LECTURE SEULE : aucun fichier déplacé, modifié ou
  supprimé — la webapp copie vers son stockage et n'y touche plus. En
  particulier, `dataset_recolte` (outil Streamlit) reste la seule source de
  vérité jusqu'à la bascule : la webapp n'y écrit JAMAIS ;
- les doublons — sha256 déjà en base, lignes actives OU archivées — sont
  signalés et ignorés sans faire échouer le lot ;
- si rien n'est importable (dossier déjà importé), refus propre : aucune
  écriture, ni en base ni dans le stockage ;
- un rapport lisible est rendu en fin d'import : créées, doublons, rejetées.

L'import crée la session, un lot par défaut « import » (backlog, découpable
ensuite), et les images au statut `en_attente_preannotation`. Les originaux
sont stockés sous `sessions/{session_id}/originals/{sha256}{ext}` : le nom par
empreinte évite tout aller-retour d'id avant insertion, et une future ligne de
recadrage partage naturellement le même fichier original.
"""
import fnmatch
import hashlib
import secrets
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

from PIL import Image as PILImage
from sqlalchemy import select

from .classes import class_count
from .models import Annotation, Batch, CaptureSession, Image, ImageStatusEvent, User
from .security import hash_password
from .storage import Storage

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_BATCH_NAME = "import"
HISTORICAL_USERNAME = "import_historique"


@dataclass
class ImportReport:
    session_name: str
    session_id: int | None = None
    batch_id: int | None = None
    created: list[str] = field(default_factory=list)
    duplicates: list[tuple[str, str]] = field(default_factory=list)  # (fichier, motif)
    rejected: list[tuple[str, str]] = field(default_factory=list)  # (fichier, motif)
    renamed: list[tuple[str, str]] = field(default_factory=list)  # (brut, export)
    aborted_reason: str | None = None

    @property
    def aborted(self) -> bool:
        return self.aborted_reason is not None

    def summary(self) -> str:
        lines = []
        if self.aborted:
            lines.append(f"IMPORT REFUSÉ — {self.aborted_reason}")
        else:
            lines.append(
                f"Import « {self.session_name} » : {len(self.created)} image(s) "
                f"créée(s) (session {self.session_id}, lot « {DEFAULT_BATCH_NAME} »)"
            )
        lines.append(f"  doublons ignorés : {len(self.duplicates)}")
        lines.extend(f"    {name} — {motif}" for name, motif in self.duplicates)
        lines.append(f"  fichiers rejetés : {len(self.rejected)}")
        lines.extend(f"    {name} — {motif}" for name, motif in self.rejected)
        lines.append(f"  renommés pour l'export : {len(self.renamed)}")
        lines.extend(f"    {brut} → {export}" for brut, export in self.renamed)
        return "\n".join(lines)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _place_original(storage: Storage, rel: str, source: Path, sha: str,
                    copied: list[str]) -> None:
    """Copie l'original, ou réutilise un fichier déjà présent au chemin sha.

    Un fichier présent est normalement identique (le nom EST l'empreinte, et
    l'écriture est atomique) — mais il peut être arrivé autrement :
    restauration de sauvegarde, copie manuelle, migration de stockage. On
    revérifie donc son empreinte complète avant de le réutiliser : le nom
    étant le sha attendu, la vérification est autovalidante, et son coût ne
    s'applique qu'au cas rare où le fichier existe déjà. Un contenu divergent
    est une corruption de stockage : on échoue explicitement, sans toucher au
    fichier. Seules les copies faites par CETTE exécution entrent dans
    `copied` (le rollback ne supprime jamais ce qu'il n'a pas créé)."""
    if storage.exists(rel):
        actual = hashlib.sha256(storage.read(rel)).hexdigest()
        if actual != sha:
            raise ValueError(
                f"stockage corrompu : {rel} a l'empreinte {actual[:12]}… au lieu "
                f"de {sha[:12]}… — à inspecter manuellement, rien n'a été écrit"
            )
        return
    storage.save_file(rel, source)
    copied.append(rel)


def _unique_export_name(name: str, label: str, taken: set[str]) -> str:
    """Nom d'export unique dans la session : intact si libre, sinon préfixé du
    dossier source, puis suffixé en dernier recours."""
    if name not in taken:
        return name
    candidate = f"{label}__{name}"
    stem, ext = Path(name).stem, Path(name).suffix
    k = 2
    while candidate in taken:
        candidate = f"{label}__{stem}_{k}{ext}"
        k += 1
    return candidate


def import_session_folder(
    db,
    storage: Storage,
    *,
    source_dirs: list[Path | str] | tuple,
    admin_id: int,
    name: str,
    captured_on: date | None = None,
    lighting: str | None = None,
    camera_height_cm: int | None = None,
    compost_state: str | None = None,
    operator: str | None = None,
    notes: str | None = None,
    attach_existing: bool = False,
) -> ImportReport:
    """Un import = UNE session, éventuellement plusieurs dossiers sources (un
    par poste de capture) : trois postes qui photographient le même tas au
    même moment doivent entrer comme une seule session, sinon le split par
    session peut mettre le poste A en entraînement et le poste B en test —
    la fuite de données qu'il doit justement empêcher.

    `attach_existing=True` rattache les images à la session `name` déjà en
    base (cas nominal d'un poste importé après coup) : les paramètres de
    session sont alors refusés — la session existante n'est pas modifiée."""
    report = ImportReport(session_name=name)
    sources = [Path(s) for s in source_dirs]
    if not sources:
        raise ValueError("aucun dossier source fourni")
    for source in sources:
        if not source.is_dir():
            raise ValueError(f"dossier source introuvable : {source}")
    existante = db.scalar(
        select(CaptureSession).where(CaptureSession.name == name))
    if attach_existing:
        if existante is None:
            raise ValueError(f"session « {name} » introuvable — pour la "
                             "créer, ne pas demander de rattachement")
        if any(v is not None for v in (captured_on, lighting, camera_height_cm,
                                       compost_state, operator, notes)):
            raise ValueError(
                "rattachement : les paramètres de session ne s'appliquent "
                "pas, la session existante n'est pas modifiée")
    else:
        # Refus AVANT toute écriture si le nom de session est pris
        if existante is not None:
            raise ValueError(f"une session « {name} » existe déjà — pour y "
                             "ajouter un poste, demander le rattachement")
        if captured_on is None:
            raise ValueError("date de capture requise pour créer une session")

    # ── Analyse des dossiers, en lecture seule ───────────────────────────────
    candidates: list[tuple[Path, str, int, int, str]] = []  # + label du poste
    seen: dict[str, str] = {}
    for source in sources:
        label = source.name
        for path in sorted(p for p in source.iterdir() if p.is_file()):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                report.rejected.append((path.name, "extension non supportée"))
                continue
            try:
                with PILImage.open(path) as im:
                    width, height = im.size
                    im.verify()
            except Exception:
                report.rejected.append((path.name, "fichier illisible"))
                continue
            sha = _sha256_file(path)
            if sha in seen:
                report.duplicates.append(
                    (path.name, f"identique à {seen[sha]}")
                )
                continue
            seen[sha] = f"{path.name} ({label})"
            candidates.append((path, sha, width, height, label))

    # Doublons déjà en base : TOUTES les lignes, actives ou archivées — une
    # image dont la seule occurrence est archivée reste un doublon (le split
    # par session serait cassé si elle réapparaissait ailleurs)
    if candidates:
        known = dict(
            db.execute(
                select(Image.sha256, Image.id).where(
                    Image.sha256.in_([c[1] for c in candidates])
                )
            ).all()
        )
        remaining = []
        for entry in candidates:
            if entry[1] in known:
                report.duplicates.append(
                    (entry[0].name, f"déjà importé (image {known[entry[1]]})")
                )
            else:
                remaining.append(entry)
        candidates = remaining

    if not candidates:
        report.aborted_reason = (
            "aucune image importable (dossier déjà importé ?) — "
            f"{len(report.duplicates)} doublon(s), {len(report.rejected)} rejet(s). "
            "Rien n'a été écrit."
        )
        return report

    # ── Écritures : base + copies, tout ou rien ──────────────────────────────
    copied: list[str] = []
    try:
        if attach_existing:
            session = existante
            batch = db.scalar(select(Batch).where(
                Batch.session_id == session.id,
                Batch.name == DEFAULT_BATCH_NAME))
        else:
            session = CaptureSession(
                name=name, captured_on=captured_on, lighting=lighting,
                camera_height_cm=camera_height_cm, compost_state=compost_state,
                operator=operator, notes=notes, created_by=admin_id,
            )
            db.add(session)
            db.flush()
            batch = None
        if batch is None:
            batch = Batch(session_id=session.id, name=DEFAULT_BATCH_NAME,
                          created_by=admin_id)
            db.add(batch)
            db.flush()

        # Noms d'export déjà pris dans la session cible (lignes actives) :
        # l'unicité (session_id, export_filename) doit survivre au rattachement
        taken: set[str] = set(db.scalars(select(Image.export_filename).where(
            Image.session_id == session.id, Image.superseded_at.is_(None))))
        for path, sha, width, height, label in candidates:
            rel = f"sessions/{session.id}/originals/{sha}{path.suffix.lower()}"
            export_name = _unique_export_name(path.name, label, taken)
            taken.add(export_name)
            if export_name != path.name:
                report.renamed.append((path.name, export_name))
            image = Image(
                session_id=session.id, batch_id=batch.id,
                original_filename=path.name, export_filename=export_name,
                source_label=label, original_path=rel,
                width=width, height=height, sha256=sha,
            )
            db.add(image)
            db.flush()
            _place_original(storage, rel, path, sha, copied)
            db.add(ImageStatusEvent(
                image_id=image.id, from_status=None,
                to_status="en_attente_preannotation", changed_by=admin_id,
            ))
            report.created.append(path.name)

        report.session_id = session.id
        report.batch_id = batch.id
        db.commit()
    except BaseException:
        db.rollback()
        for rel in copied:
            storage.delete(rel, missing_ok=True)
        raise
    return report


# ═══ Reprise historique (dataset_recolte) — à session explicite ══════════════
# Mécanisme validé le 2026-07-28, refondu le 2026-08-03 après l'aplatissement
# du dossier : le nom d'un fichier ne dit plus ni sa session ni son poste de
# façon fiable, toute reprise DÉSIGNE donc sa cible — session (créée si la
# date est fournie, rejointe sinon), poste de capture (source_label) et motif
# des fichiers concernés. Le rattachement à une session déjà importée est le
# cas nominal : deux postes qui photographient la même matière au même moment
# dans deux sessions distinctes recréeraient la fuite train/test que le split
# par session doit justement empêcher.
# Les images arrivent déjà annotées : statut `annotee`, labels YOLO convertis
# en annotations source='humain', state='validee', attribuées au compte dédié
# INACTIF `import_historique` (les vrais auteurs, par fichier, sont inconnus —
# on ne les invente pas). ABSENCE DE FICHIER LABEL = NÉGATIF (compost nu, zéro
# boîte), exactement comme un .txt vide : l'outil d'annotation n'écrit plus de
# fichier pour une image sans intrus. Les fichiers sources ne sont jamais
# renommés (dataset_recolte reste intouché). EXÉCUTION UNIQUEMENT sur ordre
# explicite.


@dataclass
class CouvertureReport:
    """Résultat de la vérification de couverture (lecture seule)."""

    par_motif: list[tuple[str, int]] = field(default_factory=list)
    orphelins: list[str] = field(default_factory=list)  # aucun motif
    recouvrements: list[tuple[str, str]] = field(default_factory=list)  # (fichier, motifs)
    labels_sans_image: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.orphelins or self.recouvrements
                    or self.labels_sans_image)

    def summary(self) -> str:
        total = sum(n for _, n in self.par_motif)
        lines = [f"Couverture : {total} fichier(s) sous "
                 f"{len(self.par_motif)} motif(s)"]
        lines.extend(f"  {motif} : {n} fichier(s)"
                     for motif, n in self.par_motif)
        lines.append(f"  hors de tout motif : {len(self.orphelins)}")
        lines.extend(f"    {n}" for n in self.orphelins)
        lines.append(f"  sous plusieurs motifs : {len(self.recouvrements)}")
        lines.extend(f"    {n} — {motifs}" for n, motifs in self.recouvrements)
        lines.append(f"  labels sans image : {len(self.labels_sans_image)}")
        lines.extend(f"    {n}" for n in self.labels_sans_image)
        lines.append(
            "PARTITION EXACTE — chaque fichier répond à un motif et un seul."
            if self.ok else
            "COUVERTURE EN ÉCHEC — à corriger avant toute reprise."
        )
        return "\n".join(lines)


def verifier_couverture(*, images_dir: Path | str, labels_dir: Path | str,
                        patterns: list[str]) -> CouvertureReport:
    """Vérifie que les motifs PARTITIONNENT le dossier d'images : chaque
    fichier doit répondre à exactement un motif. Un fichier hors de tout motif
    serait oublié en silence par les passes de reprise — le risque principal
    d'une reprise par motifs — et un fichier sous plusieurs motifs serait
    attribué au premier poste repris. Signale aussi les labels (.txt) dont
    l'image n'existe pas. Lecture seule, aucune base requise."""
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    if not images_dir.is_dir():
        raise ValueError(f"dossier images introuvable : {images_dir}")
    if not labels_dir.is_dir():
        raise ValueError(f"dossier labels introuvable : {labels_dir}")
    if not patterns:
        raise ValueError("aucun motif fourni")

    report = CouvertureReport()
    counts = dict.fromkeys(patterns, 0)  # dédoublonne en gardant l'ordre
    stems = set()
    for path in sorted(p for p in images_dir.iterdir() if p.is_file()):
        stems.add(path.stem)
        hits = [m for m in counts if fnmatch.fnmatch(path.name, m)]
        if not hits:
            report.orphelins.append(path.name)
        elif len(hits) > 1:
            report.recouvrements.append((path.name, " + ".join(hits)))
        else:
            counts[hits[0]] += 1
    report.par_motif = list(counts.items())
    report.labels_sans_image = sorted(
        p.name for p in labels_dir.iterdir()
        if p.is_file() and p.suffix == ".txt" and p.stem not in stems
    )
    return report


def _parse_label_file(path: Path, nc: int) -> list:
    """Lignes YOLO -> [(class_id, x, y, w, h)]. Tolère l'absence de retour à
    la ligne final (213 fichiers du dataset sont dans ce cas) ; un fichier
    vide est VALIDE : image sans intrus."""
    boxes = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"ligne invalide : « {raw.strip()} »")
        cid = int(parts[0])
        if not 0 <= cid < nc:
            raise ValueError(f"class_id {cid} hors référentiel (nc={nc})")
        x, y, w, h = (float(v) for v in parts[1:])
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"coordonnées hors [0,1] : « {raw.strip()} »")
        boxes.append((cid, x, y, w, h))
    return boxes


def _examiner_paire(path: Path, labels_dir: Path, nc: int, seen: dict,
                    rejected: list, duplicates: list) -> tuple | None:
    """Validation d'une paire image/label. Retourne (image, sha, largeur,
    hauteur, boîtes), ou None si la paire est rejetée ou doublon du lot
    (motif consigné dans la liste concernée). L'ABSENCE de fichier label
    n'est PAS un rejet : c'est un négatif — compost nu, zéro boîte."""
    try:
        with PILImage.open(path) as im:
            width, height = im.size
            im.verify()
    except Exception:
        rejected.append((path.name, "fichier illisible"))
        return None
    label_path = labels_dir / f"{path.stem}.txt"
    boxes: list = []
    if label_path.is_file():
        try:
            boxes = _parse_label_file(label_path, nc)
        except ValueError as exc:
            rejected.append((path.name, f"label invalide : {exc}"))
            return None
    sha = _sha256_file(path)
    if sha in seen:
        duplicates.append((path.name, f"identique à {seen[sha]}"))
        return None
    seen[sha] = path.name
    return (path, sha, width, height, boxes)


def _retirer_connus(db, entries: list, duplicates: list) -> list:
    """Écarte les empreintes déjà en base — TOUTES les lignes, actives ou
    archivées (une image dont la seule occurrence est archivée reste un
    doublon : la voir réapparaître ailleurs casserait le split par session)."""
    if not entries:
        return entries
    known = dict(
        db.execute(
            select(Image.sha256, Image.id).where(
                Image.sha256.in_([e[1] for e in entries])
            )
        ).all()
    )
    kept = []
    for entry in entries:
        if entry[1] in known:
            duplicates.append(
                (entry[0].name, f"déjà importé (image {known[entry[1]]})")
            )
        else:
            kept.append(entry)
    return kept


def _ensure_historical_user(db, admin_id: int) -> int:
    user = db.scalar(select(User).where(User.username == HISTORICAL_USERNAME))
    if user is None:
        user = User(
            username=HISTORICAL_USERNAME,
            # mot de passe aléatoire jamais communiqué : compte inactif,
            # présent uniquement pour porter la traçabilité des FK
            password_hash=hash_password(secrets.token_urlsafe(24)),
            display_name="Import historique",
            role="annotateur",
            is_active=False,
            created_by=admin_id,
        )
        db.add(user)
        db.flush()
    return user.id


def _inserer_image_annotee(db, storage: Storage, copied: list[str], *,
                           session_id: int, batch_id: int, entry: tuple,
                           export_name: str, source_label: str,
                           importer_id: int, now: datetime) -> None:
    """Insère une image reprise déjà annotée : statut `annotee`, événement de
    création, boîtes humaines validées attribuées au compte de reprise."""
    path, sha, width, height, boxes = entry
    rel = f"sessions/{session_id}/originals/{sha}{path.suffix.lower()}"
    image = Image(
        session_id=session_id, batch_id=batch_id,
        original_filename=path.name, export_filename=export_name,
        source_label=source_label, original_path=rel,
        width=width, height=height, sha256=sha,
        status="annotee", annotated_by=importer_id, annotated_at=now,
    )
    db.add(image)
    db.flush()
    _place_original(storage, rel, path, sha, copied)
    db.add(ImageStatusEvent(
        image_id=image.id, from_status=None, to_status="annotee",
        changed_by=importer_id,
    ))
    for cid, x, y, w, h in boxes:
        db.add(Annotation(
            image_id=image.id, class_id=cid,
            x_center=x, y_center=y, box_width=w, box_height=h,
            source="humain", state="validee",
            created_by=importer_id,
            decided_by=importer_id, decided_at=now,
        ))


@dataclass
class HistoricalSessionPlan:
    """Résultat de l'analyse (lecture seule) d'une reprise à session explicite."""

    session_name: str
    source_label: str
    pattern: str
    captured_on: date | None = None  # fournie = création ; None = rattachement
    session_id: int | None = None  # résolu au plan en cas de rattachement
    entries: list[tuple[Path, str, int, int, list]] = field(default_factory=list)
    hors_motif: int = 0
    duplicates: list[tuple[str, str]] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)

    @property
    def total_images(self) -> int:
        return len(self.entries)

    @property
    def total_boxes(self) -> int:
        return sum(len(e[4]) for e in self.entries)

    @property
    def negatives(self) -> int:
        """Images sans boîte — par convention (2026-08-03), l'absence de
        fichier label signifie « compost nu », pas « annotation manquante »."""
        return sum(1 for e in self.entries if not e[4])

    def summary(self) -> str:
        if self.captured_on is not None:
            cible = (f"création de la session « {self.session_name} » "
                     f"(capture du {self.captured_on.isoformat()})")
        else:
            cible = (f"rattachement à la session existante "
                     f"« {self.session_name} » (id {self.session_id})")
        lines = [
            f"Plan de reprise à session explicite : {self.total_images} "
            f"image(s), {self.total_boxes} bbox(es)",
            f"  cible : {cible}",
            f"  poste (source_label) : {self.source_label}",
            f"  négatifs (aucune boîte, compost nu) : {self.negatives}",
            f"  motif « {self.pattern} » : {self.hors_motif} fichier(s) hors "
            "motif ignoré(s) — ils relèvent d'autres passes de reprise",
            f"  doublons ignorés : {len(self.duplicates)}",
        ]
        lines.extend(f"    {n} — {m}" for n, m in self.duplicates)
        lines.append(f"  fichiers rejetés : {len(self.rejected)}")
        lines.extend(f"    {n} — {m}" for n, m in self.rejected)
        return "\n".join(lines)


def plan_historical_session(
    db,
    *,
    images_dir: Path | str,
    labels_dir: Path | str,
    pattern: str,
    session_name: str,
    source_label: str,
    captured_on: date | None = None,
) -> HistoricalSessionPlan:
    """Analyse en lecture seule — n'écrit rien, nulle part.

    Seuls les fichiers dont le NOM répond au motif glob `pattern` sont
    considérés ; les autres sont ignorés (comptés) : ils relèvent d'autres
    passes — chaque poste de capture est repris par sa propre commande, et la
    commande `couverture` garantit en amont qu'aucun fichier n'échappe à
    l'ensemble des motifs."""
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    if not images_dir.is_dir():
        raise ValueError(f"dossier images introuvable : {images_dir}")
    if not labels_dir.is_dir():
        raise ValueError(f"dossier labels introuvable : {labels_dir}")

    existing = db.scalar(
        select(CaptureSession.id).where(CaptureSession.name == session_name)
    )
    if captured_on is None and existing is None:
        raise ValueError(
            f"session « {session_name} » introuvable — pour la créer, "
            "fournir sa date de capture (--date)"
        )
    if captured_on is not None and existing is not None:
        raise ValueError(
            f"une session « {session_name} » existe déjà — pour un "
            "rattachement, ne pas fournir --date"
        )

    plan = HistoricalSessionPlan(
        session_name=session_name, source_label=source_label, pattern=pattern,
        captured_on=captured_on, session_id=existing,
    )
    nc = class_count()
    seen: dict[str, str] = {}
    for path in sorted(p for p in images_dir.iterdir() if p.is_file()):
        if not fnmatch.fnmatch(path.name, pattern):
            plan.hors_motif += 1
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            plan.rejected.append((path.name, "extension non supportée"))
            continue
        entry = _examiner_paire(path, labels_dir, nc, seen,
                                plan.rejected, plan.duplicates)
        if entry is not None:
            plan.entries.append(entry)
    plan.entries = _retirer_connus(db, plan.entries, plan.duplicates)
    return plan


def import_historical_session(db, storage: Storage, *,
                              plan: HistoricalSessionPlan,
                              admin_id: int) -> ImportReport:
    """Exécute un plan à session explicite. Tout-ou-rien : la moindre erreur
    annule tout (base ET fichiers copiés par cette exécution). Les
    vérifications de cible sont refaites : l'analyse et l'exécution sont deux
    commandes distinctes, la base a pu changer entre les deux."""
    if plan.total_images == 0:
        raise ValueError("plan vide : rien à importer (tout est doublon/rejeté ?)")

    session = db.scalar(
        select(CaptureSession).where(CaptureSession.name == plan.session_name)
    )
    if plan.captured_on is None and session is None:
        raise ValueError(f"session « {plan.session_name} » introuvable")
    if plan.captured_on is not None and session is not None:
        raise ValueError(f"une session « {plan.session_name} » existe déjà")

    report = ImportReport(session_name=plan.session_name)
    report.duplicates = list(plan.duplicates)
    report.rejected = list(plan.rejected)
    importer_id = _ensure_historical_user(db, admin_id)
    now = datetime.now(timezone.utc)
    copied: list[str] = []
    try:
        if session is None:
            session = CaptureSession(
                name=plan.session_name, captured_on=plan.captured_on,
                notes="reprise à session explicite — paramètres de capture "
                      "non renseignés, auteurs réels par image inconnus",
                created_by=admin_id,
            )
            db.add(session)
            db.flush()
        batch = db.scalar(select(Batch).where(
            Batch.session_id == session.id, Batch.name == DEFAULT_BATCH_NAME))
        if batch is None:
            batch = Batch(session_id=session.id, name=DEFAULT_BATCH_NAME,
                          created_by=admin_id)
            db.add(batch)
            db.flush()
        # Noms d'export déjà pris dans la session cible (lignes actives) :
        # l'unicité (session_id, export_filename) doit survivre au rattachement
        taken = set(db.scalars(select(Image.export_filename).where(
            Image.session_id == session.id, Image.superseded_at.is_(None))))
        for entry in plan.entries:
            export_name = _unique_export_name(entry[0].name,
                                              plan.source_label, taken)
            taken.add(export_name)
            if export_name != entry[0].name:
                report.renamed.append((entry[0].name, export_name))
            _inserer_image_annotee(
                db, storage, copied,
                session_id=session.id, batch_id=batch.id, entry=entry,
                export_name=export_name, source_label=plan.source_label,
                importer_id=importer_id, now=now,
            )
            report.created.append(entry[0].name)
        report.session_id = session.id
        report.batch_id = batch.id
        db.commit()
    except BaseException:
        db.rollback()
        for rel in copied:
            storage.delete(rel, missing_ok=True)
        raise
    return report
