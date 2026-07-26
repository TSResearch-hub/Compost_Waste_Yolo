# Modèles — versions et provenance

Les fichiers `.pt` sont **gitignorés** (trop lourds pour GitHub) : ils circulent
par le Drive du projet, dans un dossier `models/` qui reflète celui-ci. Ce
README, lui, est versionné : c'est la référence de « quel fichier est quoi ».

## Versions

| Version | Fichier | Architecture | Entraîné sur | Run source (results.csv, courbes) | Évaluation |
|---|---|---|---|---|---|
| **v0** | `v0_pretrain_rtdetr-l.pt` | RT-DETR-l | Datasets externes (préentraînement) | `runs/pretrain_16-07_03h44` | `runs/eval_pretrain_test_20-07_08h31` |
| **v0** (léger) | `v0_pretrain_yolov8n.pt` | YOLOv8n | Datasets externes (préentraînement) | — | — |
| **v1** | `v1_finetune_rtdetr-l_400captures.pt` | RT-DETR-l | v0 + ~400 captures annotées (30 epochs) | `runs/finetune_20-07_14h33` | `runs/eval_finetune_test_20-07_15h03` |
| **v2** | `v2_finetune_rtdetr-l_snapshot-v003.pt` | RT-DETR-l | v0 + snapshot v003 : 1031 images (~600 nouvelles + ~400 anciennes captures, 3 postes fusionnés), 25 epochs Colab | `runs/finetune_21-07_05h10` | `runs/eval_finetune_test_21-07_13h02` |

Métriques image (jeu de test figé `captures_test`, seuils de
`configs/alert_rules.yaml`) :

- **v1** : rappel 0.828, précision 0.889, 0.29 fausse alerte / image négative.
  Détecte 3 piles sur 4 (Composite).
- **v2** : rappel 0.931, précision 0.806, 0.48 FA / image négative.
  **Mais Composite (piles) = 0 détection** à tous les seuils — une seule pile
  dans ses données d'entraînement. Ne pas déployer seul tant que des piles
  n'ont pas été collectées et réinjectées.

## Modèle déployé

Les deux interfaces (annotation et production, dans `interfaces/`)
chargent **`weights/best.pt` à la racine du dépôt** (voir `settings.py`).
C'est actuellement une copie exacte de **v1** (même md5). Pour changer de
modèle déployé : écraser `weights/best.pt` avec le fichier de la version
choisie.

## Règles

- **Tout finetune part de v0**, jamais d'un finetune précédent : le découpage
  train/val change entre snapshots, repartir d'un finetune contaminerait le
  jeu de test et empêcherait de comparer les versions entre elles.
- **Nommage** : `vN_<rôle>_<archi>_<données>.pt` (rôle : `pretrain` ou `finetune` — les scripts le lisent pour nommer les évals). Une nouvelle version = un nouveau
  fichier ici + une ligne dans le tableau + le run correspondant conservé
  dans `runs/`.
- `v2_…` est la version **allégée** du `best.pt` de son run (optimiseur
  retiré, demi-précision : 263 → 66 Mo). Les poids d'inférence (EMA) sont
  identiques — c'est l'opération standard de fin d'entraînement Ultralytics,
  qui manque quand on rapatrie un checkpoint Colab à la main.
- `archive/` : anciens poids remplacés, gardés au cas où.

> Attention : avant le 26/07, les discussions et notes appelaient « finetune
> v2 » le modèle renommé ici **v1**, et « v3 » le modèle renommé **v2**.
