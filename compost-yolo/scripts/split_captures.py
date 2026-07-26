"""Split des captures réelles pour le fine-tuning + l'éval « compost ».

Découpe les captures (snapshot de update_dataset.py, avec images/ + labels/) en :

  - un POOL de FINE-TUNING (images + labels bruts) -> à passer à prepare_dataset.py ;
  - un TEST COMPOST mis de côté (avec data.yaml prêt) -> à passer à evaluate.py.

Le test est la SEULE mesure honnête en conditions réelles : ces images ne sont
jamais vues à l'entraînement.

APPARIEMENT PAR DISPOSITION — quand une même scène est photographiée par
plusieurs appareils (webcam ``WIN_AAAAMMJJ_HH_MM_SS`` + téléphone
``AAAAMMJJ_HHMMSS``), les deux vues sont quasi identiques : les séparer entre
train et test gonflerait artificiellement les métriques. Chaque photo téléphone
est donc rattachée à la photo webcam la plus proche dans le temps (fenêtre
--pair-window, défaut 90 s) ; l'unité de split devient la DISPOSITION : toutes
les vues d'une même scène partent du même côté. Vérification humaine :
<output>/dispositions.csv (audit) + <output>/pairs_preview.jpg (paires côte à
côte, à survoler).

Split STRATIFIÉ par session (groups.csv du snapshot fait foi) : ~--test-fraction
des images de CHAQUE session part au test, par dispositions entières.

Le pool de fine-tuning est laissé SANS groups.csv : prepare_dataset.py y fera un
split train/val PAR IMAGE (la contamination train/val n'a pas d'importance :
la seule mesure honnête est captures_test).

Usage :
    python scripts/split_captures.py --source data/captures/latest --output data/finetune

Puis (voir README) : prepare_dataset.py -> evaluate.py (avant) -> train.py ->
evaluate.py (après) — ou tout en un : scripts/retrain.py.
"""

import argparse
import csv
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
IMG_EXTS = (".jpg", ".jpeg", ".png")
WEBCAM_TIME = re.compile(r"WIN_(\d{8})_(\d{2})_(\d{2})_(\d{2})")
PHONE_TIME = re.compile(r"(?:^|_)(\d{8})_(\d{6})(?:\D|$)")


def capture_time(stem):
    """Heure de prise de vue lue dans le nom, ou None si non reconnue."""
    m = WEBCAM_TIME.search(stem)
    if m:
        return datetime.strptime("".join(m.groups()), "%Y%m%d%H%M%S")
    m = PHONE_TIME.search(stem)
    if m:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    return None


def is_webcam(stem):
    return "WIN_" in stem


def link_or_copy(src, dst):
    try:  # les snapshots sont figés : lier ne risque rien et ne copie aucun octet
        os.link(src, dst)
    except OSError:
        shutil.copy(src, dst)


def build_dispositions(images, sessions, window):
    """Groupe les images par DISPOSITION (scène physique).

    Chaque image webcam est l'ancre de sa disposition. Chaque image téléphone
    rejoint la disposition de l'image webcam la plus proche dans le temps
    (si l'écart <= window secondes), sinon forme sa propre disposition.

    Retourne (dispo_of: img -> nom de dispo, session_of_dispo, deltas: img -> s).
    """
    webcams = sorted(((capture_time(p.stem), p) for p in images if is_webcam(p.stem)
                      and capture_time(p.stem)), key=lambda tp: tp[0])
    dispo_of, session_of_dispo, deltas = {}, {}, {}
    for img in images:
        stem = img.stem
        if is_webcam(stem) or not webcams:
            dispo_of[img] = stem
            session_of_dispo.setdefault(stem, sessions.get(stem, "?"))
            continue
        t = capture_time(stem)
        if t is None:
            dispo_of[img] = stem
            session_of_dispo.setdefault(stem, sessions.get(stem, "?"))
            continue
        nearest = min(webcams, key=lambda tp: abs((tp[0] - t).total_seconds()))
        delta = abs((nearest[0] - t).total_seconds())
        if delta <= window:
            anchor = nearest[1].stem
            dispo_of[img] = anchor
            session_of_dispo.setdefault(anchor, sessions.get(anchor, "?"))
            deltas[img] = delta
        else:
            dispo_of[img] = stem
            session_of_dispo.setdefault(stem, sessions.get(stem, "?"))
    return dispo_of, session_of_dispo, deltas


def pairs_preview(dispo_images, path, thumb_h=180, cols=3):
    """Planche de vérification : pour chaque disposition multi-vues, l'ancre
    webcam et la vue téléphone côte à côte. À survoler avant de faire confiance
    au split."""
    from PIL import Image, ImageDraw, ImageFont
    multi = {d: imgs for d, imgs in dispo_images.items() if len(imgs) > 1}
    if not multi:
        return 0
    try:  # police absente sur certains environnements (Colab) : repli intégré PIL
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    cells = []
    for dispo, imgs in sorted(multi.items()):
        anchor = next((i for i in imgs if i.stem == dispo), imgs[0])
        others = [i for i in imgs if i is not anchor]
        row_imgs = [anchor] + others
        thumbs = []
        for im_path in row_imgs[:3]:  # ancre + 2 vues max par cellule
            im = Image.open(im_path).convert("RGB")
            w = int(im.size[0] * thumb_h / im.size[1])
            thumbs.append(im.resize((w, thumb_h)))
        w_total = sum(t.size[0] for t in thumbs) + 4 * (len(thumbs) - 1)
        cell = Image.new("RGB", (w_total, thumb_h + 20), (235, 235, 235))
        x = 0
        for t in thumbs:
            cell.paste(t, (x, 20))
            x += t.size[0] + 4
        d = ImageDraw.Draw(cell)
        d.text((2, 2), f"{dispo}  ({len(imgs)} vues)", font=font, fill=(20, 20, 20))
        cells.append(cell)
    cw = max(c.size[0] for c in cells)
    ch = max(c.size[1] for c in cells)
    rows = -(-len(cells) // cols)
    board = Image.new("RGB", (cols * (cw + 10) + 10, rows * (ch + 10) + 10), (235, 235, 235))
    for i, c in enumerate(cells):
        r, k = divmod(i, cols)
        board.paste(c, (10 + k * (cw + 10), 10 + r * (ch + 10)))
    board.save(path, quality=85)
    return len(multi)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="data/captures/latest",
                    help="snapshot de captures (défaut : data/captures/latest)")
    ap.add_argument("--output", default="data/finetune",
                    help="dossier de sortie (défaut : data/finetune)")
    ap.add_argument("--test-fraction", type=float, default=0.2,
                    help="part de CHAQUE session mise au test compost (défaut : 0.2)")
    ap.add_argument("--pair-window", type=float, default=90,
                    help="écart max (s) pour apparier une photo téléphone à une photo "
                         "webcam = même disposition (défaut : 90)")
    ap.add_argument("--seed", type=int, default=42, help="graine du tirage (défaut : 42)")
    ap.add_argument("--no-preview", action="store_true",
                    help="ne pas générer la planche de vérification des paires")
    args = ap.parse_args()

    random.seed(args.seed)
    src = Path(args.source)
    if not (src / "images").is_dir():
        sys.exit(f"{src}/images introuvable (attendu : un snapshot d'update_dataset.py)")

    # sessions (groups.csv du snapshot : stem -> session)
    sessions = {}
    gcsv = src / "groups.csv"
    if gcsv.exists():
        with open(gcsv, encoding="utf-8") as f:
            sessions = {r["stem"]: r["group_id"] for r in csv.DictReader(f)}

    images = sorted(p for p in (src / "images").iterdir() if p.suffix.lower() in IMG_EXTS)

    # 1. dispositions (appariement multi-vues par horodatage)
    dispo_of, session_of_dispo, deltas = build_dispositions(images, sessions, args.pair_window)
    dispo_images = defaultdict(list)
    for img, d in dispo_of.items():
        dispo_images[d].append(img)
    n_paired = sum(1 for img in images if dispo_of[img] != img.stem)

    # 2. split stratifié par session, par DISPOSITIONS entières
    by_session = defaultdict(list)  # session -> [dispo]
    for d, sess in session_of_dispo.items():
        by_session[sess].append(d)
    assign = {}  # dispo -> "finetune" | "test"
    for sess in sorted(by_session):
        dispos = sorted(by_session[sess])
        random.shuffle(dispos)
        quota = round(sum(len(dispo_images[d]) for d in dispos) * args.test_fraction)
        taken = 0
        for d in dispos:
            if taken < quota:
                assign[d] = "test"
                taken += len(dispo_images[d])
            else:
                assign[d] = "finetune"

    # 3. copie (liens) vers la sortie
    out = Path(args.output)
    ft, te = out / "captures_finetune", out / "captures_test"
    for d in (ft, te):
        shutil.rmtree(d, ignore_errors=True)
    dest = {"finetune": (ft / "images", ft / "labels"),
            "test": (te / "images" / "test", te / "labels" / "test")}
    for img_dir, lbl_dir in dest.values():
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
    for img in images:
        sub = assign[dispo_of[img]]
        img_dir, lbl_dir = dest[sub]
        link_or_copy(img, img_dir / img.name)
        lbl = src / "labels" / f"{img.stem}.txt"
        if lbl.exists():
            shutil.copy(lbl, lbl_dir / lbl.name)

    # 4. data.yaml du test + audit
    names = yaml.safe_load(open(ROOT / "configs/data.yaml", encoding="utf-8"))["names"]
    with open(te / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"path": str(te.resolve()),
                        "train": "images/test", "val": "images/test", "test": "images/test",
                        "names": dict(enumerate(names))},
                       f, allow_unicode=True, sort_keys=False)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "dispositions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stem", "disposition", "session", "split", "delta_s"])
        for img in images:
            d = dispo_of[img]
            w.writerow([img.stem, d, session_of_dispo[d], assign[d],
                        f"{deltas[img]:.0f}" if img in deltas else ""])

    # 5. récapitulatif
    counts = Counter((session_of_dispo[dispo_of[i]], assign[dispo_of[i]]) for i in images)
    order = sorted(by_session)
    for sub in ("finetune", "test"):
        n_img = sum(1 for _ in dest[sub][0].glob("*"))
        n_lbl = sum(1 for _ in dest[sub][1].glob("*.txt"))
        detail = ", ".join(f"{s}:{counts[(s, sub)]}" for s in order if counts[(s, sub)])
        print(f"  {sub:9s}: {n_img} images, {n_lbl} labels  ({detail})")
    multi = sum(1 for imgs in dispo_images.values() if len(imgs) > 1)
    print(f"  appariement : {n_paired} photo(s) téléphone rattachée(s) à "
          f"{multi} disposition(s) webcam (fenêtre {args.pair_window:.0f}s) ; "
          f"{sum(1 for i in images if not is_webcam(i.stem)) - n_paired} non appariée(s)")
    if not args.no_preview and multi:
        n = pairs_preview(dispo_images, out / "pairs_preview.jpg")
        print(f"  planche de vérification ({n} dispositions multi-vues) : "
              f"{out / 'pairs_preview.jpg'}")
    print(f"  audit : {out / 'dispositions.csv'}")
    print(f"\nPool fine-tuning -> {ft}")
    print(f"Test compost     -> {te}/data.yaml")


if __name__ == "__main__":
    main()
