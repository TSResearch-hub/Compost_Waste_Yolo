# Annotation des captures réelles

Préparation et annotation des photos prises en conditions réelles (plateau de
compost) avant de les ajouter au dataset.

> ⚠️ Détecteur d'**intrus uniquement** : on n'annote QUE les matériaux non
> compostables (Plastique, Metal, Carton, Aluminium, Ceramique). L'organique est
> le fond, on ne l'annote pas.

---

## 1. Crop des captures

On recadre sur la zone du plateau **avant** d'annoter : meilleure résolution sur
les petits intrus et suppression du décor (fûts, sol) qui ferait des faux
positifs. Trépied = cadrage fixe → **une boîte par session**.

Les previews (`grid`/`check`) sont regroupées dans
`~/stage/captures/crop_preview/` grâce à `--out`. ⚠️ Sans `--out`, l'outil écrit
dans `crop_preview/` du **dossier courant** (le repo) — d'où le `--out` partout.

Adapte `SESSION` et `IMG` à la session à traiter, puis lance les 3 commandes :

```bash
cd ~/stage/Compost_Waste_Yolo/compost-yolo && source venv/bin/activate
SESSION=~/stage/captures/session1.5            # <-- dossier de la session
PREVIEW=~/stage/captures/crop_preview          # previews regroupées ici
IMG="$SESSION"/WIN_20260618_15_48_49_Pro.jpg   # une image repère de la session

# 1. Grille de coordonnées pour LIRE la boîte
python scripts/crop_capture.py grid "$IMG" --out "$PREVIEW"/grid.jpg

# 2. Vérifier la boîte (ajuste les 4 valeurs, puis ouvre box.jpg dans Windows)
python scripts/crop_capture.py check "$IMG" --box 400 20 1505 1078 --out "$PREVIEW"/box.jpg

# 3. Appliquer à toute la session
python scripts/crop_capture.py apply "$SESSION"/ "$SESSION"_crop/ --box 400 20 1505 1078
```

Boîte `x1 y1 x2 y2` = coin haut-gauche, coin bas-droite (px). **Une boîte par
session** (le plateau change de place entre sessions) :

| Session | Plage | Boîte (provisoire) |
|---|---|---|
| session1 | 11h-12h | `180 15 1445 1065` |
| session1.5 | 14h-16h | `400 20 1505 1078` |

Règle : couvrir **tout le plateau avec une marge**, sans couper d'intrus ni
inclure le bac bleu / fût.
