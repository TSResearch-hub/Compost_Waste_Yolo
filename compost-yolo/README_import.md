# Guide pratique

## Importer et préparer un dataset externe

Objectif : ramener n'importe quel dataset au format du pipeline (labels YOLO,
ids de `configs/data.yaml`), puis le splitter. Deux commandes, une config à
écrire.

### 1. Inspecter le dataset téléchargé

```bash
unzip dataset.zip -d ~/stage/datasets/<nom>
cat ~/stage/datasets/<nom>/data.yaml                      # classes et leur ORDRE
head ~/stage/datasets/<nom>/train/labels/$(ls ~/stage/datasets/<nom>/train/labels | head -1)
```

Lignes `class_id cx cy w h` (5 nombres, valeurs 0–1) → `format: yolo`.
Lignes plus longues (`class_id x1 y1 x2 y2 ...`) → polygones de segmentation
→ `format: yolo-seg` (convertis en boîtes englobantes à l'import).

### 2. Écrire la table de correspondance `configs/import_<nom>.yaml`

```yaml
format: yolo             # ou yolo-seg si labels en polygones
source_names:            # copie EXACTE du names: source, même ordre
  - "0 rigid plastic"
  - "1 cardboard"
class_map:               # classe source -> classe de configs/data.yaml
  "0 rigid plastic": Plastique
  "1 cardboard": Carton
```

Les cas possibles dans `class_map` :

| Cas | Quoi écrire |
|---|---|
| Classe équivalente | `plastic: Plastique` |
| Fusionner 2 classes source | `rigid plastic: Plastique` et `soft plastic: Plastique` |
| Classe inutile (verre…) | `glass: null` (ou ne pas la lister) — l'image devient négative s'il ne reste rien |
| Classe à garder mais absente du projet | l'ajouter d'abord à `configs/data.yaml` (EN FIN de liste, ne jamais réordonner) |

### 3. Importer puis splitter

```bash
python scripts/import_dataset.py --source ~/stage/datasets/<nom> \
    --mapping configs/import_<nom>.yaml --output data/raw/<nom>
python scripts/prepare_dataset.py --source data/raw/<nom>
```

`import_dataset.py` réécrit les ids des labels, aplatit les noms de fichiers
et génère `groups.csv` ; `prepare_dataset.py` fusionne le split d'origine et
re-splitte en 70/15/15 par hash déterministe vers `data/processed/`.

Option d'import : `--group-by image` (défaut, images indépendantes) ou
`--group-by folder` (un dossier source = un groupe, si le dataset est organisé
par scènes dont les images se ressemblent — évite la fuite train/test).

### 4. Vérifier (3 contrôles, dans l'ordre)

1. **Récap d'import** : classes attendues > 0 instances (sinon `source_names`
   faux), taux de négatives plausible.
2. **Récap de prepare** : proportions ~70/15/15, avertissements « classe
   absente » conscients et assumés.
3. **Contrôle visuel** : `python scripts/train.py --epochs 1 --imgsz 320 --batch 8`
   puis ouvrir `runs/train_*/train_batch0.jpg` — les noms de classes doivent
   coller aux objets (seule façon de détecter un `class_map` faux).

### Cas particuliers

- **Labels en polygones de segmentation** (lignes à plus de 5 nombres) :
  mettre `format: yolo-seg` dans la config — chaque polygone est converti en
  sa boîte englobante.
- **Labels COCO (json) ou VOC (xml)** : non implémenté — écrire une fonction
  de même signature que `parse_yolo` et l'ajouter au dict `PARSERS` de
  `scripts/import_dataset.py`.
- **Dataset déjà aux classes du projet, noms `cap_<timestamp>`** (export de
  l'interface) : pas d'import, directement `prepare_dataset.py --source ...`.
- **Combiner plusieurs sources** : enchaîner les `prepare_dataset.py`, ils
  s'accumulent dans `data/processed/`. Pour repartir d'une seule source,
  vider d'abord `data/processed/images/` et `data/processed/labels/`.
- **Licence** : vérifier la clause (CC BY ok avec attribution ; CC BY-NC =
  pas d'usage commercial) et la noter en commentaire dans `import_<nom>.yaml`.
