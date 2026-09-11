"""Banc d'essai de robustesse à l'éclairage.

Applique des dégradations lumineuses contrôlées au split de test, réévalue le
modèle sur chacune, et trace mAP / AP par classe en fonction de la condition.

Répond à la question posée après la démo (« le parasol enlevé, le modèle
décroche — pourquoi ? ») sans avoir besoin de retourner sur le terrain :
les dégradations sont simulées sur des images déjà annotées, donc la vérité
terrain est inchangée et les écarts mesurés sont imputables à la seule lumière.

Familles de dégradation :
- exposition   : gamma / gain global      -> couvert par l'augmentation hsv_v
- balance      : température de couleur   -> quasi pas couvert (hsv_h = 1.5 %)
- ombre        : gradient doux, bord dur  -> pas couvert du tout
- spéculaire   : reflets brûlés (clipping)-> pas couvert du tout, irréversible
- soleil_direct: combinaison des 3 ci-dessus = scénario « parasol enlevé »

Exemple :
    python scripts/benchmark_lumiere.py \
        --weights models/v2_finetune_rtdetr-l_snapshot-v003.pt \
        --data data/finetune/captures_test_session2/data.yaml
"""

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # pas de serveur X ici
import matplotlib.pyplot as plt
import numpy as np
import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# Classes qui comptent pour la livraison (cf. cadrage Metal/Ceramique/Verre,
# Plastique si possible). Les autres sont trop peu représentées dans le test
# pour qu'une AP y soit interprétable.
CLASSES_PRIORITAIRES = ["Metal", "Ceramique", "Verre", "Plastique"]


# --------------------------------------------------------------------------
# Dégradations lumineuses
# --------------------------------------------------------------------------

def _gamma(img, g):
    """Courbe tonale. g > 1 assombrit, g < 1 éclaircit. Réversible."""
    lut = np.array([((i / 255.0) ** g) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


def _gain(img, k):
    """Gain linéaire avec écrêtage à 255 : simule la surexposition (destructif)."""
    return np.clip(img.astype(np.float32) * k, 0, 255).astype(np.uint8)


def _balance_blancs(img, r, b):
    """Décale la température de couleur. img est en BGR."""
    out = img.astype(np.float32)
    out[..., 2] *= r
    out[..., 0] *= b
    return np.clip(out, 0, 255).astype(np.uint8)


def _ombre_douce(img, force=0.40):
    """Dégradé latéral : la lumière tombe progressivement d'un côté."""
    w = img.shape[1]
    grad = np.linspace(1.0, force, w, dtype=np.float32)[None, :, None]
    return np.clip(img.astype(np.float32) * grad, 0, 255).astype(np.uint8)


def _ombre_dure(img, force=0.42, angle=22):
    """Ombre portée à bord net : crée un *bord* franc dans l'image.

    C'est le cas qui casse un détecteur, parce qu'un bord franc ressemble à
    une frontière d'objet. Aucune augmentation HSV ne produit ça.
    """
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    pente = np.tan(np.deg2rad(angle))
    masque = yy > pente * (xx - w / 2) + h / 2
    out = img.astype(np.float32)
    out[masque] *= force
    return np.clip(out, 0, 255).astype(np.uint8)


def _speculaire(img, seuil=200, force=0.9):
    """Brûle les zones déjà claires : reflets du soleil sur métal / verre.

    L'information des pixels écrêtés est définitivement perdue — c'est
    précisément ce qu'aucune augmentation ne sait simuler ni compenser.
    """
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    masque = (gris > seuil).astype(np.uint8)
    masque = cv2.dilate(masque, np.ones((9, 9), np.uint8), iterations=2)
    masque = cv2.GaussianBlur(masque.astype(np.float32), (21, 21), 0)
    masque = np.clip(masque * force, 0, 1)[..., None]
    fond = img.astype(np.float32) * (1 - masque)
    return np.clip(fond + 255 * masque, 0, 255).astype(np.uint8)


def _soleil_direct(img):
    """Scénario « parasol enlevé » : gain + reflets + lumière chaude + ombre dure."""
    out = _gain(img, 1.45)
    out = _speculaire(out, seuil=195, force=0.9)
    out = _balance_blancs(out, r=1.12, b=0.88)
    return _ombre_dure(out, force=0.50, angle=22)


# nom -> (famille, fonction, libellé lisible pour les figures)
CONDITIONS = {
    "baseline":        ("reference",  lambda im: im,                        "référence"),
    "sombre":          ("exposition", lambda im: _gamma(im, 1.6),           "sombre (γ1.6)"),
    "tres_sombre":     ("exposition", lambda im: _gamma(im, 2.2),           "très sombre (γ2.2)"),
    "clair":           ("exposition", lambda im: _gamma(im, 0.6),           "clair (γ0.6)"),
    "surexpose":       ("exposition", lambda im: _gain(im, 1.9),            "surexposé (×1.9)"),
    "wb_chaud":        ("balance",    lambda im: _balance_blancs(im, 1.18, 0.82), "lumière chaude (soleil)"),
    "wb_froid":        ("balance",    lambda im: _balance_blancs(im, 0.84, 1.20), "lumière froide (ombre)"),
    "ombre_douce":     ("ombre",      _ombre_douce,                         "ombre douce"),
    "ombre_dure":      ("ombre",      _ombre_dure,                          "ombre dure"),
    "speculaire":      ("speculaire", _speculaire,                          "reflets brûlés"),
    "soleil_direct":   ("combine",    _soleil_direct,                       "SOLEIL DIRECT (combiné)"),
}


# --------------------------------------------------------------------------
# Construction des jeux dégradés
# --------------------------------------------------------------------------

def lister_images(images_dir):
    return sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def construire_variante(images, labels_dir, sortie, fonction, imgsz, split_name):
    """Écrit une copie dégradée des images ; les labels sont liés, pas copiés."""
    img_out = sortie / "images" / split_name
    lbl_out = sortie / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for src in images:
        img = cv2.imread(str(src))
        if img is None:
            continue
        h, w = img.shape[:2]
        if max(h, w) > imgsz:  # on évalue à imgsz de toute façon
            k = imgsz / max(h, w)
            img = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(img_out / f"{src.stem}.jpg"), fonction(img),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])

        lbl_src = labels_dir / f"{src.stem}.txt"
        lbl_dst = lbl_out / f"{src.stem}.txt"
        if lbl_src.exists() and not lbl_dst.exists():
            lbl_dst.symlink_to(lbl_src.resolve())
    return img_out


def ecrire_yaml(sortie, names, split_name):
    chemin = sortie / "data.yaml"
    chemin.write_text(yaml.safe_dump({
        "path": str(sortie.resolve()),
        "train": f"images/{split_name}",
        "val": f"images/{split_name}",
        "test": f"images/{split_name}",
        "names": names,
    }, sort_keys=False, allow_unicode=True))
    return chemin


# --------------------------------------------------------------------------
# Évaluation
# --------------------------------------------------------------------------

def charger_modele(weights):
    """RT-DETR et YOLO n'ont pas la même classe d'entrée dans Ultralytics."""
    from ultralytics import RTDETR, YOLO
    if "rtdetr" in Path(weights).name.lower():
        return RTDETR(weights)
    return YOLO(weights)


def evaluer(model, data_yaml, imgsz, device, noms_par_id, dossier_runs, tag):
    res = model.val(data=str(data_yaml), split="val", imgsz=imgsz, device=device,
                    verbose=False, plots=False, save_json=False,
                    project=str(dossier_runs), name=tag, exist_ok=True)

    ligne = {"mAP50": float(res.box.map50), "mAP50-95": float(res.box.map)}
    index = getattr(res, "ap_class_index", getattr(res.box, "ap_class_index", []))
    for position, class_id in enumerate(index):
        nom = noms_par_id.get(int(class_id), str(class_id))
        p, r, ap50, ap = res.box.class_result(position)
        ligne[f"{nom}_AP50"] = float(ap50)
        ligne[f"{nom}_rappel"] = float(r)
        ligne[f"{nom}_precision"] = float(p)
    return ligne


# --------------------------------------------------------------------------
# Sorties
# --------------------------------------------------------------------------

def tracer(resultats, noms_classes, sortie_png):
    ordre = [c for c in CONDITIONS if c in resultats]
    libelles = [CONDITIONS[c][2] for c in ordre]
    base = resultats["baseline"]["mAP50"] if "baseline" in resultats else None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10))

    couleurs = {"reference": "#444444", "exposition": "#4C78A8", "balance": "#F58518",
                "ombre": "#E45756", "speculaire": "#B279A2", "combine": "#000000"}
    valeurs = [resultats[c]["mAP50"] for c in ordre]
    ax1.bar(libelles, valeurs, color=[couleurs[CONDITIONS[c][0]] for c in ordre])
    if base is not None:
        ax1.axhline(base, ls="--", lw=1, color="#444444")
        ax1.text(len(ordre) - 0.4, base, " référence", va="bottom", fontsize=9, color="#444444")
    for i, v in enumerate(valeurs):
        perte = "" if base in (None, 0) else f"\n{(v - base) / base * 100:+.0f}%"
        ax1.text(i, v, f"{v:.3f}{perte}", ha="center", va="bottom", fontsize=8)
    ax1.set_ylabel("mAP50 (toutes classes)")
    ax1.set_title("Robustesse du modèle à la dégradation lumineuse", fontweight="bold")
    ax1.set_ylim(0, max(valeurs) * 1.28)
    ax1.tick_params(axis="x", rotation=30)
    for lab in ax1.get_xticklabels():
        lab.set_ha("right")

    presentes = [c for c in CLASSES_PRIORITAIRES
                 if any(f"{c}_AP50" in resultats[k] for k in ordre)]
    largeur = 0.8 / max(len(presentes), 1)
    x = np.arange(len(ordre))
    for i, classe in enumerate(presentes):
        ys = [resultats[c].get(f"{classe}_AP50", np.nan) for c in ordre]
        ax2.bar(x + i * largeur - 0.4 + largeur / 2, ys, largeur, label=classe)
    ax2.set_xticks(x)
    ax2.set_xticklabels(libelles, rotation=30, ha="right")
    ax2.set_ylabel("AP50 par classe")
    ax2.set_title("Classes prioritaires de la livraison", fontweight="bold")
    ax2.legend(ncol=len(presentes))
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(sortie_png, dpi=140)
    print(f"\nFigure  -> {sortie_png}")


def ecrire_csv(resultats, sortie_csv):
    colonnes = ["condition", "famille"]
    for ligne in resultats.values():
        for c in ligne:
            if c not in colonnes:
                colonnes.append(c)
    with open(sortie_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=colonnes)
        w.writeheader()
        for nom in CONDITIONS:
            if nom in resultats:
                w.writerow({"condition": nom, "famille": CONDITIONS[nom][0], **resultats[nom]})
    print(f"CSV     -> {sortie_csv}")


def resumer(resultats):
    if "baseline" not in resultats:
        return
    base = resultats["baseline"]
    print("\n" + "=" * 78)
    print(f"{'condition':<26}{'mAP50':>9}{'Δ vs réf':>11}   {'famille':<12}")
    print("-" * 78)
    for nom in CONDITIONS:
        if nom not in resultats:
            continue
        v = resultats[nom]["mAP50"]
        d = (v - base["mAP50"]) / base["mAP50"] * 100 if base["mAP50"] else 0
        print(f"{CONDITIONS[nom][2]:<26}{v:>9.3f}{d:>10.1f}%   {CONDITIONS[nom][0]:<12}")
    print("=" * 78)


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True, help="data.yaml du jeu de test annoté")
    ap.add_argument("--split", default="test", help="sous-dossier images/<split>")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--limit", type=int, default=0, help="n images max (test rapide)")
    ap.add_argument("--work", default=None, help="dossier de travail des variantes")
    ap.add_argument("--out", default=None, help="dossier des résultats")
    ap.add_argument("--conditions", default="", help="liste séparée par des virgules")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.data).read_text())
    racine = Path(cfg.get("path", Path(args.data).parent))
    images_dir = racine / "images" / args.split
    labels_dir = racine / "labels" / args.split
    if not images_dir.is_dir():
        raise SystemExit(f"Introuvable : {images_dir}")

    noms = cfg["names"]
    noms_par_id = {int(k): v for k, v in noms.items()} if isinstance(noms, dict) \
        else dict(enumerate(noms))

    images = lister_images(images_dir)
    if args.limit:
        images = images[:args.limit]
    print(f"{len(images)} images de test  |  labels : {labels_dir}")

    choisies = [c.strip() for c in args.conditions.split(",") if c.strip()] or list(CONDITIONS)
    inconnues = [c for c in choisies if c not in CONDITIONS]
    if inconnues:
        raise SystemExit(f"Condition(s) inconnue(s) : {inconnues}")

    work = Path(args.work or "/tmp/claude-1000/-home-aliikched-stage/"
                "56e58756-8c0c-49af-8bc6-b0c478bdc6a1/scratchpad/bench_lumiere")
    out = Path(args.out or Path(args.weights).parent.parent / "runs" / "benchmark_lumiere")
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    print("Chargement du modèle…")
    model = charger_modele(args.weights)

    resultats = {}
    for i, nom in enumerate(choisies, 1):
        famille, fonction, libelle = CONDITIONS[nom]
        print(f"\n[{i}/{len(choisies)}] {libelle}  ({famille})")
        dossier = work / nom
        if dossier.exists():
            shutil.rmtree(dossier)
        construire_variante(images, labels_dir, dossier, fonction, args.imgsz, args.split)
        yaml_variante = ecrire_yaml(dossier, noms, args.split)
        resultats[nom] = evaluer(model, yaml_variante, args.imgsz, args.device,
                                 noms_par_id, out / "_val", nom)
        print(f"    mAP50 = {resultats[nom]['mAP50']:.3f}")

        apercu = cv2.imread(str(next((dossier / "images" / args.split).glob("*.jpg"))))
        cv2.imwrite(str(out / f"apercu_{nom}.jpg"), apercu)

    resumer(resultats)
    ecrire_csv(resultats, out / "benchmark_lumiere.csv")
    tracer(resultats, list(noms_par_id.values()), out / "benchmark_lumiere.png")
    print(f"Aperçus -> {out}/apercu_*.jpg")


if __name__ == "__main__":
    main()
