# Interface de production sur la Jetson

La Jetson fait tourner **l'onglet Production uniquement** : surveillance caméra,
alertes, captures à annoter. L'interface est servie sur le réseau local — on la
pilote depuis n'importe quel navigateur (PC, téléphone) sur le même réseau.

Le réentraînement et l'évaluation restent sur Colab ou un PC avec GPU : les
8 Go de la Jetson sont partagés entre CPU et GPU, c'est insuffisant.

## 1. Prérequis

Un environnement Python où l'inférence Ultralytics fonctionne déjà sur la
Jetson (torch NVIDIA + TensorRT + OpenCV). Vérifier :

```bash
python -c "import torch; print(torch.cuda.is_available())"   # doit répondre True
python -c "from ultralytics import YOLO; print('ok')"
```

Si ce n'est pas le cas, suivre le guide officiel :
https://docs.ultralytics.com/guides/nvidia-jetson/

## 2. Installer le projet

Cloner le dépôt, puis installer **uniquement** le paquet du projet et
Streamlit :

```bash
cd Compost_Waste_Yolo
pip install -e . --no-deps
pip install streamlit
```

**Ne PAS lancer `pip install -r requirements.txt` sur la Jetson** : il
réinstallerait torch et OpenCV en versions génériques, ce qui casserait les
versions NVIDIA spécifiques à la Jetson.

## 3. Exporter le modèle en TensorRT (une fois par modèle déployé)

Le modèle déployé `weights/best.pt` arrive avec le dépôt. TensorRT le compile
en un moteur optimisé pour le GPU de la Jetson — l'inférence est plusieurs
fois plus rapide qu'avec le `.pt` :

```bash
python scripts/export.py --weights weights/best.pt --formats engine --half
```

L'export prend plusieurs minutes et crée `weights/best.engine`. Ce fichier
n'est valable **que sur cette Jetson** (il dépend du GPU et de la version de
TensorRT) : ne pas le copier ailleurs, et le régénérer après chaque
mise à jour de `weights/best.pt`.

## 4. Lancer l'interface

```bash
streamlit run interfaces/production/finetune_app.py --server.address 0.0.0.0 --server.port 8502
```

Puis ouvrir :

- sur la Jetson : http://localhost:8502
- depuis un autre appareil du réseau : http://«IP-de-la-Jetson»:8502
  (obtenir l'IP avec `hostname -I`)

## 5. LED d'alerte et bouton de capture (optionnel)

Sur le connecteur 40 broches de la Jetson (numéros = position physique de la
broche, la broche 1 est marquée sur la carte) :

| Élément | Câblage |
|---------|---------|
| LED     | broche **12** → résistance 220-330 Ω → LED (patte longue) → GND (broche 14) |
| Bouton  | broche **18** → bouton poussoir → 3,3 V (broche 17) |

- La **LED s'allume tant qu'une alerte est en cours** et s'éteint quand la
  scène redevient saine.
- Le **bouton déclenche une capture manuelle** : même effet que le bouton
  « Capture manuelle » de l'interface — à utiliser quand l'employé voit un
  intrus que le modèle ne signale pas. L'image brute part dans `a_annoter/`.

Les broches se changent dans `configs/hardware.yaml` (commentaires de câblage
inclus). Il faut la bibliothèque GPIO (souvent déjà présente avec JetPack) :

```bash
python -c "import Jetson.GPIO"        # si erreur :
pip install Jetson.GPIO
sudo usermod -aG gpio $USER           # puis rouvrir la session
```

Sans GPIO (PC, WSL, Docker), l'interface fonctionne normalement — LED et
bouton sont simplement ignorés. Quand ils sont actifs, la ligne d'état sous
la vidéo affiche « LED + bouton GPIO actifs ».

## 6. Utiliser

Dans l'onglet **Production** :

1. Modèle : choisir **« déployé (TensorRT) — weights/best.engine »**
   (proposé en premier quand il existe).
2. Caméra : laisser `0` (la caméra USB). L'application la configure
   automatiquement en MJPEG 1920×1080 pour un flux fluide.
3. « Démarrer la surveillance ».

Les captures (alertes, manuelles, saines) s'enregistrent comme partout
ailleurs dans `runs/production_*/` — les images brutes de `a_annoter/` sont à
donner comme source à l'interface d'annotation.
