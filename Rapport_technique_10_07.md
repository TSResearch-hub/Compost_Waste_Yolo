# Rapport technique — 10/07 : fiabilisation du référentiel de classes + boucle d'entraînement complète

**Trois volets :** (1) correction de deux bugs invisibles qui polluaient les données,
(2) fermeture de la boucle « annoter → réentraîner » (onglet Dataset, export, `train.py`),
(3) mise sous tests automatisés (32 tests).

---

## Volet 1 — Deux bugs invisibles corrigés (qualité des données)

### 🐛 Bug n°1 : les pré-annotations Metal / Céramique / Verre / Composite arrivaient classées « Aluminium »

Le modèle actuel (`weights/best.pt`) nomme ses classes **sans accents** et n'en a que 7 :

| Modèle (7 classes) | Référentiel du projet (9 classes) | Correspondance avant | Après |
|---|---|---|---|
| `Plastique` | `Plastique` (0) | ✅ 0 | ✅ 0 |
| `Metal` | `Métal` (1) | ❌ **Aluminium (3)** | ✅ 1 |
| `Carton` | `Carton` (2) | ✅ 2 | ✅ 2 |
| `Aluminium` | `Aluminium` (3) | ✅ 3 | ✅ 3 |
| `Ceramique` | `Céramique` (4) | ❌ **Aluminium (3)** | ✅ 4 |
| `Verre` | `Verre` (7) | ❌ **Aluminium (3)** | ✅ 7 |
| `Composite` | `Composite` (8) | ❌ **Aluminium (3)** | ✅ 8 |

Cause : `CLASS_MAP.get(nom, 3)` — tout nom inconnu (accent manquant) retombait
**silencieusement** sur la classe 3. Concrètement : chaque pré-annotation IA d'un
métal, d'une céramique, d'un verre ou d'un composite arrivait dans l'éditeur
étiquetée « Aluminium », à recorriger à la main… ou pas (erreurs passées dans le
dataset). Impact PC **et** mobile (même fonction).

**Correctif** : comparaison normalisée (minuscules, sans accents) via
`normalize_class_name()` / `class_name_to_id()` dans `helper.py` ; une classe
réellement inconnue est désormais signalée en console au lieu d'être mal classée
en silence.

**Conséquence sur le dataset existant : les 17 bboxes « Aluminium » sont suspectes**
(certaines sont probablement des Metal/Céramique/Verre mal reclassés). Un filtre
« 🏷 Aluminium » a été ajouté dans l'onglet Vérification pour les repasser en revue
rapidement (voir volet 2).

### 🐛 Bug n°2 : les alertes de la détection en direct étaient inversées

Dans `settings.py` traînaient deux alias hérités (`RECYCLABLE = NON_COMPOSTABLE`,
`NON_RECYCLABLE = COMPOSTABLE`) consommés dans le mauvais ordre par l'affichage :

- un **plastique** détecté s'affichait **« ✔ COMPOSTABLE » en vert** ;
- un déchet **organique** s'affichait « ✖ NON-COMPOSTABLE » ;
- et à cause des accents (bug n°1), **Metal et Ceramique ne déclenchaient aucune
  alerte du tout**.

Pour un outil dont la finalité est justement d'alerter sur les intrus du compost,
c'était le pire comportement possible — et il était invisible tant qu'on ne
comparait pas l'alerte à l'objet posé devant la caméra.

**Correctif** : `classify_waste_type()` matche directement les listes
`COMPOSTABLE / NON_COMPOSTABLE / MATIERE_RISQUEE / DANGEREUX` (accents normalisés),
alias supprimés. Vérifié par tests.

### 🐛 Bonus : flux webcam déformé + chemins relatifs

- Le flux de détection forçait **640×360 (16:9)** : une caméra 4:3 était déformée à
  l'écran **et dans les frames capturées pour annotation** (donc dans le dataset).
  → redimensionnement à ratio conservé (`resize_keep_ratio`).
- `SAVE_DIR` et le CSV de chronométrage étaient encore relatifs au répertoire de
  lancement → basés sur `settings.ROOT` comme le reste (cohérent avec le correctif
  du 09/07).

---

## Volet 2 — La boucle « annoter → réentraîner » est fermée

Jusqu'ici l'outil produisait un dataset, mais le réentraînement (objectif affiché
du projet) restait manuel. Trois ajouts :

### 📊 Nouvel onglet « Dataset »

- **Répartition des classes** : barres aux couleurs des bboxes de l'éditeur,
  effectifs et pourcentages écrits en clair. Sur le dataset actuel :

  | Classe | Bboxes | | Classe | Bboxes |
  |---|---|---|---|---|
  | Métal | 203 (54 %) | | Composite | 16 |
  | Plastique | 77 | | Carton | 4 |
  | Céramique | 35 | | **Organique** | **0** |
  | Verre | 24 | | **Papier** | **0** |
  | Aluminium | 17 ⚠ | | | |

  L'onglet signale explicitement les classes à 0 : **le prochain modèle ne saura
  pas les détecter** (question pour la réunion : est-ce voulu ?).
- **Analyse d'anomalies** (à la demande) : labels orphelins, lignes illisibles,
  classes hors référentiel, **bboxes fantômes/hors cadre**, **doublons de contenu**
  (même photo sous deux noms = fuite train→val possible). Première exécution sur le
  dataset réel : **1 bbox fantôme trouvée** (`20260618_121740.txt`, 0.0017×0.0013 —
  résidu de l'ancien bug de clic corrigé le 09/07).
- **Export d'entraînement** : un bouton génère `exports/export_<date>/` au format
  YOLO (`images/train|val`, `labels/train|val`, `data.yaml` avec les 9 classes).
  Le **split est stratifié par classe rare** : un split purement aléatoire pouvait
  laisser Carton (4 bboxes) totalement absent de la validation. Part de validation
  réglable, seed fixe (reproductible).

### 🏋️ `train.py`

```bash
python train.py --data exports/export_XXXX/data.yaml            # défauts raisonnables
python train.py --data ... --epochs 50 --model yolo11n.pt       # variantes
```

Fine-tuning depuis `weights/best.pt` par défaut (ultralytics adapte la tête 7→9
classes automatiquement), device auto (GPU si dispo), arrêt anticipé, sorties dans
`runs/`, et la commande de déploiement (`cp ... weights/best.pt`) affichée à la fin.
La commande exacte est aussi affichée dans l'onglet Dataset après chaque export.

### 🔍 Vérification : corbeille + filtre par classe

- **« 🗑 Supprimer »** (piste listée au rapport du 09/07) : avec confirmation,
  l'image et son label partent dans `dataset_recolte/corbeille/` — **réversible**,
  rien n'est effacé ; trace `status=deleted` dans le CSV ; passage auto à l'image
  suivante.
- **Filtre « contient la classe X »** : sert notamment à repasser sur les 17
  « Aluminium » suspects du bug n°1.

---

## Volet 3 — Mise sous tests (32 tests, ~13 s)

`python -m pytest tests/` (pytest + httpx ajoutés aux requirements) :

- **Référentiel** : les 7 noms réels du modèle → bons ids ; fallback signalé ;
  conversion centre→coin des pré-annotations.
- **Alertes** : plastique bien non-compostable, organique compostable, noms sans
  accents du modèle bien routés (verrouille les deux bugs corrigés).
- **API mobile de bout en bout** (vrai modèle, dataset temporaire) : save + fichier
  YOLO exact, validation avant écriture (pas d'image orpheline), ré-annotation même
  contenu → même nom, collision → `_2`, galerie/label, traversée de chemin refusée.
- **Outils dataset** : distribution, chaque type d'anomalie, corbeille (y compris
  collisions), split stratifié (classe rare présente en val), déterminisme,
  `data.yaml` accents inclus.

Vérifications complémentaires : `streamlit.testing.AppTest` exécute réellement
`app.py` (0 exception, 5 onglets, 376 bboxes comptées sur le vrai dataset, filtre
12 options, export présent).

---

## À décider en réunion

1. **Repasser les « Aluminium »** : filtre 🏷 Aluminium dans Vérification (17 bboxes,
   certaines probablement mal classées par l'ancien bug).
2. **Corriger la bbox fantôme** signalée par l'analyse (`20260618_121740`).
3. **Organique / Papier à 0** : on n'annote que les contaminants (choix assumé) ou
   il faut collecter ces classes ? Impact direct sur ce que le prochain modèle saura faire.
4. **Lancer le premier réentraînement** sur les 233 images (`train.py`) — le modèle
   actuel n'a jamais vu nos données terrain ; sur CPU prévoir plusieurs heures ou
   zipper un export pour Colab.

## Fichiers

- **Modifiés** : `helper.py`, `settings.py`, `app.py`, `annotation_timer.py`,
  `requirements.txt`, `.gitignore`, `README.md`.
- **Créés** : `dataset_tools.py` (module pur, testable), `train.py`,
  `tests/{conftest,test_helper,test_dataset_tools,test_mobile_api}.py`.
- **Aucun rebuild npm nécessaire** (composant React inchangé) ; front mobile inchangé
  (il profite des correctifs via `helper.py` côté serveur).
