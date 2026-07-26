# Version mobile — annotation sur tablette / smartphone (PWA)

Application web tactile pour annoter le dataset depuis un téléphone ou une
tablette. Le mobile n'est qu'un **écran tactile** : le modèle YOLO et le
dataset restent sur le PC, le téléphone s'y connecte en WiFi.

```
┌─────────────┐   photo (WiFi local)   ┌───────────────────────────┐
│  Téléphone   │ ─────────────────────▶ │  PC : FastAPI + YOLO      │
│  (navigateur)│ ◀───────────────────── │  → dataset_recolte/       │
└─────────────┘   bboxes pré-annotées   └───────────────────────────┘
```

## Lancement

```bash
# depuis la racine du projet, avec le venv activé
pip install -r requirements.txt        # fastapi/uvicorn inclus
python interfaces/mobile/server.py                # port 8000 par défaut
```

Le terminal affiche l'adresse à ouvrir **sur le mobile** (même réseau WiFi) :

```
  Sur ce PC     : http://localhost:8000
  Sur le mobile : http://192.168.x.x:8000
```

> Sous Windows, autoriser Python dans le pare-feu à la première ouverture.
> Sous WSL2, lancer plutôt le serveur depuis Python Windows, ou rediriger le
> port (`netsh interface portproxy`).

## Fonctionnalités

- **📷 Prendre une photo** : capture caméra → pré-annotation IA → correction
  au doigt → sauvegarde dans `dataset_recolte/` (format YOLO identique au PC).
- **🖼 Importer des images** : annotation en série depuis la galerie du téléphone.
- **🔍 Vérifier le dataset** : galerie de miniatures du dataset (filtre
  annotées / sans annotation), correction en place des labels existants.
- **Éditeur tactile** : 1 doigt = dessiner / déplacer / redimensionner
  (poignées), 2 doigts = zoom + déplacement de la vue, double-tap = recadrer.
  Undo/redo, suppression, changement de classe, surlignage, chrono par image.
- Les durées d'annotation sont journalisées dans le **même CSV** que la
  version PC (`annotation_times.csv`, source `mobile`).

## PWA (installation sur l'écran d'accueil)

L'application embarque un manifest + service worker. En HTTP sur le réseau
local elle fonctionne comme un site web classique (c'est l'usage normal).
L'installation "app" (icône sur l'écran d'accueil, plein écran) nécessite un
contexte sécurisé — sur Android : menu Chrome → « Ajouter à l'écran
d'accueil » fonctionne aussi en HTTP local.

## Fichiers

```
mobile/
├── server.py              # FastAPI : /api/predict, /api/save, /api/gallery,
│                          # /api/thumb, /api/label, /api/stats + statiques
└── static/
    ├── index.html         # 3 vues : accueil, galerie, éditeur
    ├── app.js             # éditeur canvas tactile (vanilla JS, sans build)
    ├── style.css
    ├── manifest.webmanifest
    ├── sw.js              # cache de l'app shell
    └── icons/
```
