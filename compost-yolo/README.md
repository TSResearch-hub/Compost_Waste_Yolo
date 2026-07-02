# compost-yolo — entraînement YOLO pour le tri du compost

Repo d'entraînement et d'évaluation d'un modèle YOLO (Ultralytics) qui détecte
des matériaux intrus dans du compost filmé par caméra. La classification est
**orientée matériau** : on annote la matière (`Plastique`, `Metal`...), pas
l'objet. Cible de déploiement : Raspberry Pi 4 (CPU).

Les classes sont définies à UN SEUL endroit : [`configs/data.yaml`](configs/data.yaml).

## Installation

Python 3.10+ requis.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -e ".[dev]"
```

## Pipeline en 5 commandes

```bash
# 0. (optionnel) Importer un dataset externe — voir README_import.md
python scripts/import_dataset.py --source ~/stage/datasets/taco \
    --mapping configs/import_taco.yaml --output data/raw/taco

# 1. Préparer le dataset (split train/val/test PAR GROUPE/SESSION)
python scripts/prepare_dataset.py --source data/raw/mon_export

# 2. Entraîner (hyperparamètres dans configs/train_default.yaml, surchargeables)
python scripts/train.py --epochs 100

# 3. Évaluer (mAP Ultralytics + métriques custom par classe et niveau image)
python scripts/evaluate.py --weights runs/train_xxx/weights/best.pt --split test

# 4. Inférer sur une image / un dossier / une vidéo, avec logique d'alerte
python scripts/infer.py --weights runs/train_xxx/weights/best.pt \
    --source photo.jpg --alert-rules configs/alert_rules.yaml

# 5. Exporter pour le Raspberry Pi (ONNX + NCNN)
python scripts/export.py --weights runs/train_xxx/weights/best.pt
```

Chaque script affiche son aide détaillée avec `--help`.

## Fine-tuning sur les captures réelles + éval « compost »

Stratégie recommandée : **pré-entraîner** sur les datasets externes (beaucoup
d'images, mais autre domaine), puis **fine-tuner** sur les captures réelles (le
domaine de déploiement). L'évaluation honnête se fait sur un jeu de captures
**mis de côté** — jamais vu à l'entraînement. On y mesure le modèle **avant**
(éval B) puis **après** fine-tuning (éval C), avec le *même* `evaluate.py`.

Chaque étape est un script séparé :

```bash
# 1. SPLIT — pool de fine-tuning + test compost held-out (stratifié par session)
python scripts/split_captures.py --source data/raw/captures --output data/finetune

# 2. PRÉPARER le pool de fine-tuning (split train/val PAR IMAGE, pas de test interne :
#    le test compost, c'est data/finetune/captures_test)
python scripts/prepare_dataset.py --source data/finetune/captures_finetune \
    --output data/finetune/dataset_finetune --ratios 0.85 0.15 0

# 3. ÉVAL B — le modèle PRÉ-ENTRAÎNÉ sur le test compost (référence avant fine-tuning)
python scripts/evaluate.py --weights chemin/vers/pretrain_best.pt \
    --data data/finetune/captures_test/data.yaml --split test

# 4. FINE-TUNER — on repart du pré-entraîné (--model) avec un learning rate bas (--lr0)
python scripts/train.py --model chemin/vers/pretrain_best.pt \
    --data data/finetune/dataset_finetune/data.yaml --epochs 30 --lr0 0.001

# 5. ÉVAL C — le modèle FINE-TUNÉ sur le MÊME test compost (après)
python scripts/evaluate.py --weights runs/train_xxx/weights/best.pt \
    --data data/finetune/captures_test/data.yaml --split test
```

Compare les deux `eval_*` (B vs C) : le fine-tuning a-t-il amélioré la détection
sur le vrai compost ? `split_captures.py` affiche les commandes 2→5 avec les bons
chemins à la fin de son exécution. Sur GPU local, ajoute `--device 0` (et
`--batch 4` si mémoire insuffisante) aux étapes 3-5. Le pré-entraînement (étape 0)
se fait sur Colab (voir plus bas) ou en local avec `train.py` sur les datasets externes.

## Le point critique : split PAR SESSION

Les images d'une même session de capture (même compost, même éclairage) sont
quasi identiques. Un split aléatoire par image mettrait des quasi-doublons en
train ET en test → métriques faussement bonnes. `prepare_dataset.py` répartit
donc des **sessions entières** entre train/val/test.

Les noms de fichiers (`cap_{timestamp_unix}.jpg`) ne contiennent pas
d'identifiant de session, donc les sessions sont reconstruites :

1. si un manifeste `sessions.csv` (`session_id,start_ts,end_ts`) est présent
   dans le dossier source, il fait foi ;
2. sinon, clustering temporel : deux captures séparées de plus de
   `--session-gap-minutes` (défaut 60) = deux sessions différentes.

L'affectation d'une session à un split est un **hash déterministe de son id**
(+ `--seed`) : ajouter de nouvelles sessions ne change jamais le split des
anciennes (métriques comparables entre versions du dataset, pas de
contamination train→test).

Le nommage `cap_{timestamp}` n'est pas obligatoire : un `groups.csv`
(`stem,group_id`) dans le dossier source fixe les groupes explicitement, et
les fichiers aux noms libres sans `groups.csv` forment chacun leur propre
groupe (split par image).

## Importer un dataset externe

Pour tester d'autres datasets (Kaggle, Roboflow...), `import_dataset.py` les
ramène au format attendu par `prepare_dataset.py` : copie des images,
réécriture des labels avec les ids de classes de `configs/data.yaml` (via une
table de correspondance YAML, modèle dans
[`configs/import_example.yaml`](configs/import_example.yaml)) et génération du
`groups.csv` :

```bash
python scripts/import_dataset.py --source ~/datasets/trashnet \
    --mapping configs/import_trashnet.yaml --output data/raw/trashnet
python scripts/prepare_dataset.py --source data/raw/trashnet
```

`--group-by image` (défaut) répartit les images indépendamment ;
`--group-by folder` garde chaque dossier source dans le même split (utile si
le dataset est organisé par scène/batch). Formats de labels gérés : `yolo`
(boîtes) et `yolo-seg` (polygones de segmentation, convertis en boîtes
englobantes) ; pour COCO/VOC, ajouter une fonction dans le dict `PARSERS` du
script.

Relancer `prepare_dataset.py` avec une autre source **accumule** les datasets
dans `data/processed/` (le split par hash garantit qu'une image déjà splittée
ne change jamais de split) ; vider `data/processed/images` et `labels` pour
repartir d'une seule source.

Guide pas à pas (cas possibles, vérifications) : [README_import.md](README_import.md).

## Échanges avec le repo de l'interface d'annotation

**Entrée — récupérer les données annotées.** L'interface (repo
`Compost_Waste_Yolo`) exporte les captures dans son dossier `dataset_recolte/`.
Copier cet export vers `data/raw/` de ce repo :

```bash
cp -r ../dataset_recolte data/raw/export_$(date +%Y_%m_%d)
python scripts/prepare_dataset.py --source data/raw/export_$(date +%Y_%m_%d)
```

**Sortie — déployer le modèle vers l'interface.** Le `best.pt` d'un
entraînement remplace le modèle de pré-annotation de l'interface
(`train.py` affiche le chemin exact en fin de run) :

```bash
cp runs/train_xxx/weights/best.pt ../weights/best.pt
```

## Entraînement sur Google Colab (GPU)

Le code se développe et se versionne en local (sous-dossier `compost-yolo/`
du repo `Compost_Waste_Yolo`) ; les entraînements réels tournent sur Colab.
**Le code se modifie dans le repo et se commit, jamais dans le notebook.**
**Colab est en lecture seule vis-à-vis de git** : la session clone le repo et
n'y commit ni push jamais (aucune cellule du notebook n'écrit vers git) ; les
résultats partent vers Drive, pas vers git.

Workflow ([notebooks/colab_train.ipynb](notebooks/colab_train.ipynb)) :

1. zipper le dataset **prêt pour prepare** (export `cap_*` de l'interface, ou
   sortie `data/raw/<nom>` de `import_dataset.py` : `images/`, `labels/`,
   `groups.csv` à la racine du zip) et le déposer sur Drive
   (`MyDrive/compost/dataset_raw.zip`, un zip par dataset) ;
2. créer un token GitHub **personnel classique** (Settings → Developer
   settings → Tokens (classic), scope `repo`) et le stocker dans les Secrets
   Colab sous le nom `GITHUB_TOKEN` — il ne sert qu'au clone ;
3. exécuter les cellules : clone (lecture seule), `pip install`, montage
   Drive, copie + dézippage du dataset vers `/content/` (jamais
   d'entraînement directement sur le Drive monté, trop lent),
   `prepare_dataset.py` (une fois par dataset, ils s'accumulent), `train.py`,
   puis copie du run vers Drive.

Cycle de travail : modifier le code en local → `pytest` → commit + push →
la session Colab suivante clone automatiquement la dernière version.

Les sessions Colab peuvent sauter : `train.py --backup-dir` copie les
checkpoints vers Drive toutes les N epochs. Pour reprendre après un crash
(la VM est vidée : il faut d'abord ré-exécuter les cellules clone / install /
dataset / montage Drive, puis restaurer le backup avant `--resume`) :

```python
# retrouver le nom du run sauvegardé (ex. train_02-07_10h15)
!ls /content/drive/MyDrive/compost/backups

# restaurer le backup -> /content/runs, puis reprendre là où il s'était arrêté
RUN = "train_02-07_10h15"   # <-- le nom affiché ci-dessus
!mkdir -p /content/runs
!cp -r /content/drive/MyDrive/compost/backups/{RUN} /content/runs/
!python scripts/train.py --resume /content/runs/{RUN}/weights/last.pt \
    --backup-dir /content/drive/MyDrive/compost/backups --backup-every 10
```

Garder `--backup-dir` à la reprise, sinon la suite du run n'est plus sauvegardée.
Inutile de repasser `--data`/`--runs-dir` : le `last.pt` contient déjà toute la
config du run.

## Configuration

| Fichier | Rôle |
|---|---|
| [configs/data.yaml](configs/data.yaml) | **Seul** endroit où les classes sont définies |
| [configs/train_default.yaml](configs/train_default.yaml) | Hyperparamètres d'entraînement (modèle, epochs, seed...) |
| [configs/alert_rules.yaml](configs/alert_rules.yaml) | Classes intruses + seuil de confiance par classe |

Pour calibrer les seuils d'alerte : `evaluate.py --sweep-thresholds` exporte
les courbes rappel/précision niveau image en CSV.

## Tests

```bash
pytest
```

Un mini-dataset factice est fourni dans `tests/fixtures/mini_dataset/` pour
essayer `prepare_dataset.py` sans vraies données :

```bash
python scripts/prepare_dataset.py --source tests/fixtures/mini_dataset --output /tmp/mini
```

## Structure

```
configs/            # classes, hyperparamètres, règles d'alerte
data/raw/           # exports bruts de l'interface + datasets importés (gitignoré)
data/processed/     # dataset au format Ultralytics, généré (gitignoré)
scripts/            # import_dataset, prepare_dataset, train, evaluate, infer, export
src/compost_detection/  # logique testée : split par session, métriques, alerte
tests/              # pytest
notebooks/          # orchestration Colab (aucune logique métier)
runs/               # sorties d'entraînement/évaluation (gitignoré)
```
