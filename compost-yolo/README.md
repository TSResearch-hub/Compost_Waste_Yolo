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

## Boucle de réentraînement : annotations → dataset → fine-tuning (2 commandes)

Stratégie : **pré-entraîner** sur les datasets externes (beaucoup d'images,
autre domaine — sur Colab, voir plus bas), puis **fine-tuner** sur les captures
réelles. L'évaluation honnête se fait sur un jeu de captures **mis de côté**,
jamais vu à l'entraînement, mesuré **avant** et **après** fine-tuning.

Prérequis (une fois) : déposer le modèle pré-entraîné dans `models/`
(ex. `models/pretrain_yolov8n.pt`) — c'est le point de départ **canonique** de
chaque réentraînement (jamais le fine-tuné précédent : le split peut changer
quand le dataset grandit, seul un départ du pré-entraîné garantit que le test
n'a jamais été appris).

```bash
# 1. Intégrer les dernières annotations de l'interface (../dataset_recolte par défaut).
#    Crée un SNAPSHOT figé data/captures/vNNN_<date> (le précédent + les nouveautés) :
#    fusion multi-postes, déduplication par CONTENU d'image (md5), labels modifiés
#    mis à jour, session = date du nom de fichier. Les snapshots ne bougent jamais
#    (entraînements reproductibles) ; data/captures/latest pointe le dernier.
python scripts/update_dataset.py                         # ou --source poste1/ poste2/ ...

# 2. Réentraîner sur le dernier snapshot : split (test compost préservé) -> éval AVANT
#    -> fine-tuning -> éval APRÈS -> comparaison. --deploy copie le best.pt vers l'interface.
python scripts/retrain.py                                # options : --epochs, --batch 4, --deploy
```

### Interface graphique (pour non-initiés)

Les mêmes actions en interface web — mise à jour du dataset, histogrammes,
réentraînement, évaluation, résultats :

```bash
pip install -e ".[app]"          # une fois (installe streamlit)
streamlit run finetune_app.py
```

L'app n'a aucune logique propre : chaque bouton appelle les scripts ci-dessus
(journal affiché en direct). L'onglet **Évaluer** permet aussi de mesurer le
modèle sur une **nouvelle session avant de l'intégrer** au dataset (mesure de
généralisation honnête).

Les runs sont nommés par rôle : `runs/pretrain_*` (datasets externes),
`runs/finetune_*` (captures), `runs/eval_pretrain_*` / `runs/eval_finetune_*`
(évaluations, le JSON contient le chemin exact des poids évalués).

Chaque étape reste un script utilisable seul (`split_captures.py`,
`prepare_dataset.py`, `train.py --model ... --lr0 0.001 --run-prefix finetune`,
`evaluate.py`) : `retrain.py` ne fait que les enchaîner — voir son `--help`.

### RT-DETR (alternative à YOLO)

Tous les scripts acceptent indifféremment des poids YOLO ou RT-DETR (Ultralytics
détecte l'architecture au chargement). Pour comparer : pré-entraîner avec
`MODEL = 'rtdetr-l.pt'` dans `colab_train.ipynb` (modèle ~10× plus gros que
yolov8n : réduire `--batch`), déposer le résultat dans
`models/pretrain_rtdetr-l.pt`, puis `python scripts/retrain.py --pretrain
models/pretrain_rtdetr-l.pt --batch 4`. Les deux modèles sont alors évalués sur
le **même** test compost — comparaison directe.

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
dataset / montage Drive, puis relancer avec `--resume`) :

```python
# retrouver le nom du run sauvegardé (ex. train_02-07_10h15)
!ls /content/drive/MyDrive/compost/backups

# reprendre directement depuis le backup Drive (adapter le nom du run)
!python scripts/train.py \
    --resume /content/drive/MyDrive/compost/backups/train_02-07_10h15/weights/last.pt \
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
finetune_app.py     # interface web de réentraînement (streamlit run finetune_app.py)
data/raw/           # imports bruts (datasets externes, ancien dataset captures) (gitignoré)
data/captures/      # snapshots FIGÉS du dataset de captures : vNNN_<date> + latest (gitignoré)
data/processed/     # dataset externe au format Ultralytics, généré (gitignoré)
data/finetune/      # dossier de travail du réentraînement, généré (gitignoré)
models/             # modèles pré-entraînés canoniques, ex. pretrain_yolov8n.pt (gitignoré)
scripts/            # update_dataset, retrain + import_dataset, split_captures,
                    # prepare_dataset, train, evaluate, infer, export, crop_capture
src/compost_detection/  # logique testée : split par session, métriques, alerte, nommage
tests/              # pytest
notebooks/          # orchestration Colab/local (aucune logique métier)
runs/               # pretrain_*, finetune_*, eval_* (gitignoré)
```
