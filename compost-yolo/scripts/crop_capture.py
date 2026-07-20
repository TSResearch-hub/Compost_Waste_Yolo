"""Outil de crop des captures réelles (par session, trépied = cadrage fixe).

Workflow (sans interface graphique, compatible WSL) :

Les previews (grid/check) sont écrites dans crop_preview/ (dossier dédié, pas
mélangé aux photos sources).

1. GRILLE — affiche une grille de coordonnées sur une image pour LIRE la boîte :
       python scripts/crop_capture.py grid "img.jpg"
   Ouvre crop_preview/<nom>_grid.jpg dans Windows et lis x1,y1 (coin haut-gauche)
   et x2,y2 (coin bas-droite) de la zone du plateau.

2. VÉRIF — dessine la boîte choisie pour confirmer avant d'appliquer :
       python scripts/crop_capture.py check "img.jpg" --box 480 120 1450 1050

3. APPLIQUER — croppe TOUT un dossier (une session) avec cette boîte :
       python scripts/crop_capture.py apply dossier_in/ dossier_out/ --box 480 120 1450 1050

Conseil : une boîte par session (le plateau peut changer de place entre sessions).
Garde une petite marge pour ne pas couper un intrus posé au bord.
"""

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

EXTS = (".jpg", ".jpeg", ".png")


def list_images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in EXTS)


def cmd_grid(args):
    im = Image.open(args.image).convert("RGB")
    w, h = im.size
    d = ImageDraw.Draw(im)
    step = args.step
    for x in range(0, w, step):
        d.line([(x, 0), (x, h)], fill=(255, 0, 0), width=1)
        d.text((x + 2, 2), str(x), fill=(255, 255, 0))
    for y in range(0, h, step):
        d.line([(0, y), (w, y)], fill=(255, 0, 0), width=1)
        d.text((2, y + 2), str(y), fill=(255, 255, 0))
    out = args.out or f"crop_preview/{Path(args.image).stem}_grid.jpg"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    im.save(out)
    print(f"Image : {w}x{h} px")
    print(f"Grille (pas {step}px) -> {out}")
    print("Ouvre-la dans Windows, lis x1 y1 (haut-gauche) et x2 y2 (bas-droite) du plateau.")


def cmd_check(args):
    im = Image.open(args.image).convert("RGB")
    x1, y1, x2, y2 = args.box
    d = ImageDraw.Draw(im)
    d.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=6)
    out = args.out or f"crop_preview/{Path(args.image).stem}_box.jpg"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    im.save(out)
    print(f"Boîte {args.box} dessinée -> {out}")
    print("Vérifie qu'elle entoure bien le plateau (avec marge) sans couper d'intrus.")


def cmd_apply(args):
    x1, y1, x2, y2 = args.box
    src, dst = Path(args.in_dir), Path(args.out_dir)
    dst.mkdir(parents=True, exist_ok=True)
    imgs = list_images(src)
    if not imgs:
        sys.exit(f"Aucune image trouvée dans {src}")
    for p in imgs:
        Image.open(p).convert("RGB").crop((x1, y1, x2, y2)).save(dst / p.name)
    print(f"{len(imgs)} image(s) croppées ({x2 - x1}x{y2 - y1}) -> {dst}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("grid", help="grille de coordonnées pour lire la boîte")
    g.add_argument("image")
    g.add_argument("--step", type=int, default=100, help="espacement de la grille (px)")
    g.add_argument("--out")
    g.set_defaults(func=cmd_grid)

    c = sub.add_parser("check", help="dessine la boîte pour la vérifier")
    c.add_argument("image")
    c.add_argument("--box", type=int, nargs=4, required=True, metavar=("X1", "Y1", "X2", "Y2"))
    c.add_argument("--out")
    c.set_defaults(func=cmd_check)

    a = sub.add_parser("apply", help="croppe tout un dossier")
    a.add_argument("in_dir")
    a.add_argument("out_dir")
    a.add_argument("--box", type=int, nargs=4, required=True, metavar=("X1", "Y1", "X2", "Y2"))
    a.set_defaults(func=cmd_apply)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
