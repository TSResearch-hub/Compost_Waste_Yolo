"""Export YOLO depuis la base (lot L3) — forme validée le 2026-07-31.

Sortie consommable telle quelle par compost-yolo/scripts/prepare_dataset.py,
sans y toucher :

    sortie/
      images/       fichiers nommés d'après export_filename (stem dédoublonné)
      labels/       un .txt par image, même stem — boîtes state='validee' seules
      groups.csv    stem,group_id — group_id = ID de session, immuable par
                    construction : renommer une session ne doit JAMAIS pouvoir
                    déplacer son split (le préfixe de collision, lui, reste
                    basé sur le nom : il n'influence pas le split)
      classes.txt   référentiel utilisé à l'export (le pipeline l'ignore) —
                    un export retrouvé dans six mois reste interprétable même
                    si data.yaml a évolué entre-temps

Règles :
- périmètre : statut annotee|relue, superseded_at IS NULL ;
- le fichier image exporté est celui auquel les coordonnées se rapportent :
  le crop s'il existe, l'original sinon (le trigger CW001 garantit que toutes
  les boîtes d'une image sont relatives au même fichier) ;
- image sans intrus : .txt VIDE (0 octet), toujours présent — c'est un
  exemple négatif, pas une annotation manquante ;
- fichiers non vides : chaque ligne terminée par \n, dernière comprise ;
- collision de stem entre sessions : préfixe « nom-de-session__ » uniquement
  en collision réelle, renommages listés au rapport ;
- class_id hors référentiel data.yaml : échec explicite, jamais de label faux ;
- tout-ou-rien : construction dans <sortie>.part puis renommage — un échec
  (fichier de stockage manquant compris) ne laisse RIEN dans la sortie ;
- lecture seule stricte sur le stockage et la base.
"""
import csv
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select

from .classes import class_names
from .models import Annotation, CaptureSession, Image
from .storage import Storage


@dataclass
class ExportReport:
    output_dir: str
    # une entrée par session sélectionnée : (id, nom, images, boîtes) —
    # une session à zéro se voit d'un coup d'œil
    sessions: list[tuple[int, str, int, int]] = field(default_factory=list)
    images: int = 0
    boxes: int = 0
    empty_labels: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)
    renamed: list[tuple[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Export YOLO : {self.images} image(s), {self.boxes} boîte(s) "
            f"validée(s), {self.empty_labels} label(s) vide(s) → {self.output_dir}",
            f"  sessions couvertes : {len(self.sessions)}",
        ]
        lines.extend(
            f"    [id {sid}] {name} : {n_img} image(s), {n_box} boîte(s)"
            for sid, name, n_img, n_box in self.sessions
        )
        lines.append("  par classe :")
        lines.extend(f"    {name} : {n}" for name, n in self.class_counts.items())
        lines.append(f"  renommés pour l'export : {len(self.renamed)}")
        lines.extend(f"    {avant} → {apres}" for avant, apres in self.renamed)
        return "\n".join(lines)


def _stem_unique(stem: str, session_name: str, taken: set[str]) -> str:
    """Stem unique dans l'export : intact si libre, sinon préfixé du nom de
    session, suffixé en dernier recours (même logique que l'import)."""
    if stem not in taken:
        return stem
    candidate = f"{session_name}__{stem}"
    k = 2
    while candidate in taken:
        candidate = f"{session_name}__{stem}_{k}"
        k += 1
    return candidate


def export_yolo(db, storage: Storage, *, output_dir: Path | str,
                session_names: list[str] | None = None) -> ExportReport:
    """Exporte une session, plusieurs (par nom), ou tout (None)."""
    output = Path(output_dir)
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise ValueError(
                f"répertoire de sortie non vide : {output} — l'export exige "
                "un répertoire vide ou inexistant, rien n'a été écrit")

    # ── Sessions sélectionnées ───────────────────────────────────────────────
    rows = db.execute(
        select(CaptureSession.id, CaptureSession.name)
        .order_by(CaptureSession.id)).all()
    if session_names is not None:
        connues = {name: sid for sid, name in rows}
        absentes = [n for n in session_names if n not in connues]
        if absentes:
            raise ValueError(
                "session(s) inconnue(s) : " + ", ".join(absentes))
        rows = [(sid, name) for sid, name in rows if name in set(session_names)]
    if not rows:
        raise ValueError("aucune session à exporter")
    nom_de = dict(rows)
    session_ids = list(nom_de)

    # ── Lecture : images exportables puis boîtes validées ────────────────────
    images = db.execute(
        select(Image.id, Image.session_id, Image.export_filename,
               Image.cropped_path, Image.original_path)
        .where(Image.session_id.in_(session_ids),
               Image.status.in_(("annotee", "relue")),
               Image.superseded_at.is_(None))
        .order_by(Image.session_id, Image.id)).all()

    boites: dict[int, list] = defaultdict(list)
    if images:
        for image_id, cid, x, y, w, h in db.execute(
            select(Annotation.image_id, Annotation.class_id,
                   Annotation.x_center, Annotation.y_center,
                   Annotation.box_width, Annotation.box_height)
            .where(Annotation.image_id.in_([i[0] for i in images]),
                   Annotation.state == "validee")
            .order_by(Annotation.image_id, Annotation.id)):
            boites[image_id].append((cid, x, y, w, h))

    # Référentiel : validé AVANT toute écriture — jamais de label faux
    noms_classes = class_names()
    nc = len(noms_classes)
    for image_id, lot in boites.items():
        for cid, *_ in lot:
            if not 0 <= cid < nc:
                raise ValueError(
                    f"class_id {cid} hors référentiel (nc={nc}) sur "
                    f"l'image {image_id} — le data.yaml a-t-il changé depuis "
                    "l'annotation ? Export annulé, rien n'a été écrit")

    # ── Noms de sortie (collisions inter-sessions) ───────────────────────────
    plan = []  # (stem final, extension, chemin stockage, session_id, boîtes)
    taken: set[str] = set()
    report = ExportReport(output_dir=str(output))
    for image_id, session_id, export_filename, cropped, original in images:
        rel = cropped or original
        stem = _stem_unique(Path(export_filename).stem,
                            nom_de[session_id], taken)
        taken.add(stem)
        final = f"{stem}{Path(rel).suffix}"
        if stem != Path(export_filename).stem:
            report.renamed.append((export_filename, final))
        plan.append((stem, final, rel, session_id, boites[image_id]))

    # ── Écriture tout-ou-rien : <sortie>.part puis bascule ───────────────────
    part = output.parent / (output.name + ".part")
    if part.exists():
        shutil.rmtree(part)  # reste d'un échec précédent : à nous, jetable
    (part / "images").mkdir(parents=True)
    (part / "labels").mkdir()
    try:
        for stem, final, rel, _sid, lot in plan:
            try:
                data = storage.read(rel)
            except FileNotFoundError:
                raise ValueError(
                    f"fichier manquant dans le stockage : {rel} — export "
                    "annulé, rien n'a été écrit") from None
            (part / "images" / final).write_bytes(data)
            # non vide : chaque ligne finit par \n ; vide : 0 octet
            (part / "labels" / f"{stem}.txt").write_text(
                "".join(f"{cid} {x!r} {y!r} {w!r} {h!r}\n"
                        for cid, x, y, w, h in lot),
                encoding="utf-8")
        with open(part / "groups.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["stem", "group_id"])
            writer.writerows((stem, sid) for stem, _f, _r, sid, _l in plan)
        (part / "classes.txt").write_text(
            "".join(f"{n}\n" for n in noms_classes), encoding="utf-8")
        if output.exists():
            output.rmdir()  # vide, vérifié en entrée
        os.replace(part, output)
    except BaseException:
        shutil.rmtree(part, ignore_errors=True)
        raise

    # ── Rapport ──────────────────────────────────────────────────────────────
    par_session: dict[int, list[int]] = {sid: [0, 0] for sid in session_ids}
    report.class_counts = {n: 0 for n in noms_classes}
    for _stem, _final, _rel, sid, lot in plan:
        par_session[sid][0] += 1
        par_session[sid][1] += len(lot)
        if lot:
            for cid, *_ in lot:
                report.class_counts[noms_classes[cid]] += 1
        else:
            report.empty_labels += 1
    report.sessions = [(sid, nom_de[sid], n_img, n_box)
                       for sid, (n_img, n_box) in par_session.items()]
    report.images = len(plan)
    report.boxes = sum(len(lot) for *_x, lot in plan)
    return report
