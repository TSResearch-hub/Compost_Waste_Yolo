# webapp — annotation multi-comptes (socle)

Portage web multi-comptes de l'outil d'annotation : FastAPI + PostgreSQL +
SQLAlchemy + Alembic. Ce dossier est autonome — il ne modifie rien du reste du
repo et n'importe aucun de ses modules ; il ne partage que `data.yaml`
(référentiel de classes) et, à terme, les poids.

## Ce que garantit la base elle-même

- **Pré-annotation jamais exportable** : une boîte du modèle naît
  `state='proposee'` ; l'export ne prendra que `state='validee'` sur des images
  `annotee`/`relue`, et le passage à `annotee` exige zéro `proposee` restante.
- **Verrou de lot** : au plus un annotateur actif par lot (index unique partiel
  sur `batch_assignments`), historique des assignations inclus.
- **Machine à états** : transitions de statut d'image en liste blanche
  (trigger, SQLSTATE `CW002`), journal complet dans `image_status_events`.
- **Gel du crop** : géométrie de crop immuable dès la première annotation
  (trigger, SQLSTATE `CW001`) ; un recadrage crée une nouvelle ligne `images`
  (archivage `superseded_at`/`superseded_by_id`).
- **Split par session** : l'image porte son `session_id` jusqu'à l'export, et
  une FK composite garantit que lot et image sont dans la même session.
- **Pas de double import** : `sha256` unique sur les lignes actives ; l'import
  déduplique contre toutes les lignes (actives et archivées), signale et
  continue.

Les classes ne vivent QUE dans `data.yaml` (variable `DATA_YAML_PATH`) : la
base ne stocke que des index ; l'ordre du `data.yaml` ne doit jamais changer
par insertion au milieu.

## Installation (dev)

Venv **dédié** à webapp (isolé du venv de l'outil Streamlit existant : une
montée de version ici ne doit pas pouvoir le casser) :

```bash
cd webapp
python -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env   # puis remplir — .env n'est JAMAIS commité
```

La configuration passe uniquement par variables d'environnement (ou
`webapp/.env` en dev) : voir `.env.example`. PostgreSQL est requis — la
concurrence worker/annotateur ne se valide pas sur autre chose.

### Base de dev, par ordre de préférence

1. **Instance hébergée** (Neon/Supabase, offre gratuite) — données de test
   uniquement, jamais de données réelles. Renseigner `DATABASE_URL`.
2. **PostgreSQL portable sans droits admin** (utilisé pour valider ce socle,
   WSL2 sans sudo) : binaires zonky extraits en espace utilisateur —

   ```bash
   curl -sSL -o pg16.jar 'https://repo1.maven.org/maven2/io/zonky/test/postgres/embedded-postgres-binaries-linux-amd64/16.4.0/embedded-postgres-binaries-linux-amd64-16.4.0.jar'
   python -c "import zipfile; zipfile.ZipFile('pg16.jar').extract('postgres-linux-x86_64.txz')"
   mkdir -p pg16 && tar -xJf postgres-linux-x86_64.txz -C pg16
   pg16/bin/initdb -D pgdata -U compost_test -A trust -E UTF8 --no-sync
   pg16/bin/pg_ctl -D pgdata -l pg.log \
     -o "-p 55432 -c listen_addresses=127.0.0.1 -c unix_socket_directories='/tmp'" start
   # DATABASE_URL=postgresql+psycopg://compost_test@127.0.0.1:55432/postgres
   ```

## Migrations

```bash
cd webapp
alembic upgrade head      # applique le schéma
alembic check             # vérifie modèles ↔ base (zéro dérive attendue)
```

La migration `0001` est intégralement réversible (`alembic downgrade base`).
Les triggers `CW001`/`CW002` y sont écrits à la main ; la liste des transitions
autorisées existe en deux exemplaires (`models.IMAGE_STATUS_TRANSITIONS` et le
trigger) — toute évolution doit toucher les deux.

## Premier administrateur, puis comptes

```bash
cd webapp
.venv/bin/python -m app.cli create-admin --username reda
```

Le mot de passe est demandé interactivement (`--password-env VAR` pour les
scripts). La commande refuse s'il existe déjà un administrateur actif : les
comptes suivants se créent via l'API (`POST /api/users`), connecté en
administrateur — pas d'auto-inscription.

Lancement : `.venv/bin/uvicorn app.main:app` (depuis `webapp/`).
Endpoints du socle : `POST /api/auth/login`, `POST /api/auth/logout`,
`GET /api/auth/me`, `POST|GET /api/users`, `PATCH /api/users/{id}`,
`GET /api/health`. Session par cookie `HttpOnly`, expiration glissante par
inactivité (`AUTH_INACTIVITY_MINUTES`) tenue côté serveur.

## Import d'un lot

Depuis un dossier déposé côté serveur, **lu et jamais modifié** — la webapp
copie vers `STORAGE_ROOT` et ne touche plus à la source. `dataset_recolte`
(outil Streamlit) reste la seule source de vérité jusqu'à la bascule : la
webapp n'y écrit jamais ; la bascule sera franche, sans double écriture.

```bash
.venv/bin/python -m app.cli import-session \
  --source /depot/2026-07-18/posteA --source /depot/2026-07-18/posteB \
  --name 2026-07-18_tas-A --date 2026-07-18 \
  --admin reda [--lighting naturel] [--camera-height-cm 120] \
  [--compost-state frais] [--operator Hamza] [--notes ...]
```

`--source` est **répétable : plusieurs dossiers (un par poste de capture) =
UNE seule session**. Des postes qui photographient le même tas au même moment
ne doivent jamais entrer comme des sessions distinctes : le split par session
pourrait mettre le poste A en entraînement et le poste B en test — la fuite
de données qu'il doit empêcher. Les noms de fichiers en collision entre
postes sont désambiguïsés dans `export_filename` (préfixe du dossier source,
uniquement en cas de collision réelle ; unicité garantie par la base par
session) ; `original_filename` garde le nom brut, jamais modifié.

Équivalent API : `POST /api/imports` (administrateur, `source_dirs` liste).
Comportement :
- doublons (sha256 déjà en base, lignes actives **ou archivées**, ou fichier
  en double dans le dossier) : signalés et ignorés, le lot continue ;
- dossier déjà importé (rien d'importable) : refus propre, **aucune écriture**
  (409 côté API, code de sortie 1 côté CLI) ;
- fichiers illisibles ou d'extension non supportée : rejetés et listés ;
- rapport final : créées / doublons / rejetées, avec les motifs ;
- tout-ou-rien : si une copie échoue en cours de route, base et stockage
  reviennent à l'état initial.

L'import crée la session, un lot par défaut « import » (backlog à découper en
lots de N via l'écran admin à venir) et les images en
`en_attente_preannotation`, chacune avec son événement de traçabilité.

## Assignation, verrou, découpage

- `POST /api/batches/{id}/assign` (admin) : assigne un lot à un compte actif.
  Le verrou est l'index unique partiel de `batch_assignments` — sous deux
  assignations simultanées, PostgreSQL n'en laisse passer qu'une (testé avec
  deux transactions concurrentes réelles). Refus 409 avec le nom du détenteur.
- `POST /api/batches/{id}/release` : le détenteur rend le lot (`rendu`,
  défaut, ou `termine`) ; un admin libère de force (`force_admin`). À la
  libération, les images `en_cours` redescendent à `a_annoter`, transitions
  tracées — le repreneur continue là où ça s'est arrêté.
- `POST /api/batches/{id}/split` (admin) : découpe le backlog « import » en
  lots de N (`lot 1`, `lot 2`, … — numérotation qui saute les noms pris).
  Les images sont réparties en **round-robin entre les dossiers sources**
  (postes de capture) : un lot homogène par poste corrélerait le style d'un
  annotateur à un appareil. Seules les images non entamées et non archivées
  bougent ; refus si le lot est verrouillé.
- `GET /api/batches` : liste avec session, détenteur actif et avancement.

## Worker de pré-annotation

Processus séparé du serveur web : il ne communique avec l'application que par
la base et le stockage (en **lecture seule** — il ne crée, ne modifie et ne
supprime aucun fichier), ce qui permettra de le déplacer sur une machine GPU
sans toucher au serveur.

Venv **séparé** (torch est lourd, le serveur web n'en veut pas) :

```bash
cd webapp
python -m venv .venv-worker
.venv-worker/bin/pip install -r requirements-worker.txt
.venv-worker/bin/python -m app.worker            # boucle continue
.venv-worker/bin/python -m app.worker run --une-passe   # vide la file et s'arrête
```

Variables (voir `.env.example`) : `WEIGHTS_PATH` (obligatoire),
`WORKER_CONF_THRESHOLD` (0.10), `WORKER_MAX_DET` (20 — le max historique
observé est 9 boîtes/image), `WORKER_POLL_SECONDS`, `WORKER_MAX_ATTEMPTS`,
`WORKER_MAX_CONSECUTIVE_FAILURES`.

**Compatibilité des poids** : au chargement, les noms de classes des poids
doivent être un **préfixe exact** du `data.yaml` (ordre et orthographe) —
tout renommage, réordonnancement ou classe surnuméraire est refusé, le worker
ne démarre pas. Accepter un préfixe plutôt qu'une égalité est un
assouplissement **transitoire** (validé le 2026-07-31) : le `best.pt` actuel
sort 7 classes et la 8ᵉ, Eponge, n'a aucun exemple annoté — des poids
réentraînés en 8 classes ne la détecteraient pas davantage, exiger l'égalité
bloquerait la file pour rien. Les classes hors de portée sont journalisées
nommément au démarrage (« classe « Eponge » … ne sera JAMAIS proposée ») ;
la garde redevient une équivalence d'elle-même dès qu'un réentraînement
produit toutes les classes.

Comportement :
- **une image = une transaction** : réclamation par `FOR UPDATE SKIP LOCKED`
  (deux workers ne traitent jamais la même image ; un crash rollback et
  l'image revient dans la file), inférence, boîtes `modele`/`proposee` avec
  `confidence` et `model_name`, transition `a_annoter` tracée (`changed_by`
  NULL : le système) — le tout committé ensemble ;
- coordonnées relatives au **fichier annoté** : le crop s'il existe, sinon
  l'original ;
- reprise : les boîtes `modele`/`proposee` existantes sont écrasées (jamais
  touchées par un humain, par définition) ; toute autre annotation sur une
  image en attente = invariant violé, image garée sans y toucher ;
- **échec** : l'image reste `en_attente_preannotation`, compteur + motif +
  catégorie (`fichier_illisible`, `moteur_indisponible`, `invariant_viole`).
  Au plafond de tentatives elle est **« garée »** : hors de la file, comptée
  à part dans `GET /api/batches` (`parked_images`) — une file qui ne progresse
  plus se voit. Déblocage : remettre `preannotation_attempts` à 0 (retenter)
  ou passer l'image en `a_annoter` à la main (écran admin à venir) ;
- garde systémique : N échecs consécutifs ⇒ arrêt en erreur (code 2) — des
  poids cassés ne doivent pas garer toute la file ;
- `model_name` = `poids@sha256-court`, **rechargé si le fichier change**
  (mtime/taille) : remplacer `best.pt` sans redémarrer ne fait pas mentir la
  traçabilité ;
- arrêt propre sur SIGINT/SIGTERM : l'image en cours se termine, le worker ne
  repart pas sur la suivante.

**Base hébergée** : la transaction reste volontairement ouverte pendant
l'inférence (c'est le verrou). Le worker pose donc
`idle_in_transaction_session_timeout = '15min'` sur ses sessions (paramètre
modifiable par session) et journalise la valeur du serveur au démarrage —
vérifier cette ligne au premier lancement :
`SELECT reset_val FROM pg_settings WHERE name = 'idle_in_transaction_session_timeout';`

### Calibration du seuil — avant la première file complète

```bash
.venv-worker/bin/python -m app.worker essai --echantillon 30 \
  --seuils 0.05,0.10,0.25,0.40
```

Lecture seule (aucune écriture, aucun statut modifié) : une inférence par
image au seuil minimal avec plafond large, puis la distribution du nombre de
boîtes par image est rapportée pour chaque seuil — on calibre
`WORKER_CONF_THRESHOLD` et `WORKER_MAX_DET` sur des chiffres, pas sur un
principe. L'échantillon est **stratifié** : tiré à parts égales sur chaque
session (les conditions de capture diffèrent, le seuil doit tenir partout),
dans un ordre pseudo-aléatoire déterministe qui mêle les postes. Les images
déjà annotées sont préférées à la file d'attente : la vérité terrain permet
de séparer images **porteuses** de boîtes et **négatifs** (compost nu) — sur
un négatif, toute détection est fausse, et c'est le pire cas pour
l'annotateur, qui doit tout supprimer sans rien garder.

**Limite de la mesure : elle juge le VOLUME, pas la pertinence.** Compter
les boîtes borne la charge de suppression de l'annotateur ; un compte
proposé égal au compte réel sur les porteuses ne prouve PAS que les
propositions recouvrent les vraies boîtes — deux boîtes « au bon nombre »
peuvent être deux boîtes fausses ailleurs dans l'image. Juger la pertinence
exige un appariement IoU contre la vérité terrain : chantier à ouvrir avant
de conclure que la pré-annotation fait gagner du temps.

## Import historique (dataset_recolte) — exécution sur ordre uniquement

Depuis l'aplatissement du dossier (2026-08-03 : tout dans `images/` +
`labels/`, plus de sous-dossiers par poste), le nom d'un fichier ne dit plus
ni sa session ni son poste de façon fiable. **Toute reprise est donc à
session explicite** : une commande par poste de capture, qui désigne la
session cible, le poste et le motif des fichiers. La reprise par jour
(sessions déduites du préfixe `AAAAMMJJ_`) a été supprimée — ses axiomes
« dossier parent = poste » et « un jour = une session à un poste » sont morts
avec l'aplatissement.

**Convention forte — absence de fichier label = négatif.** Une image sans
`.txt` est du compost nu, **pas** une annotation manquante : l'outil
d'annotation n'écrit plus de fichier vide. Elle est importée en statut
`annotee` avec zéro boîte, exactement comme un `.txt` vide auparavant, et
ressort de l'export en `.txt` vide de 0 octet. Le plan de reprise compte ces
négatifs séparément : une proportion aberrante sur un poste doit alerter.

### 1. Couverture — avant toute reprise

```bash
.venv/bin/python -m app.cli couverture \
  --images .../dataset_recolte/images --labels .../dataset_recolte/labels \
  --motif "captures_1_images_WIN_20260618_*" \
  --motif "captures_1.5_images_WIN_20260618_*" \
  --motif "20260618_*" \
  --motif "20260714_*_ali.jpg" --motif "20260714_*_hamza.jpg" \
  --motif "IMG_*_reda.jpg"
```

Lecture seule. Vérifie que les motifs **partitionnent** `images/` : tout
fichier hors de tout motif, ou sous plusieurs motifs, est listé et la
commande échoue (code 1) — un fichier qu'aucun motif ne couvre serait oublié
en silence par les passes de reprise, c'est le risque principal de la
manœuvre. Signale aussi les labels sans image. À relancer à chaque mise à
jour du dataset.

### 2. Reprise, poste par poste

```bash
.venv/bin/python -m app.cli import-historique \
  --images .../images --labels .../labels --admin reda \
  --session session_2026-06-18 --date 2026-06-18 \
  --poste webcam_avant_pause --motif "captures_1_images_WIN_20260618_*"
# ANALYSE SEULE par défaut ; --execute pour écrire, sur ordre uniquement
```

`--session` + `--date` : **création** de la session ; `--session` sans
`--date` : **rattachement** à une session existante — le cas nominal, car
plusieurs postes qui photographient la même matière au même moment doivent
entrer dans UNE session (deux sessions pour la même matière recréeraient la
fuite train/test que le split par session empêche). `--poste` renseigne
`source_label` (le round-robin du découpage mêle alors les postes),
`--motif` borne les fichiers de la passe (les autres relèvent d'autres
passes). Le dataset actuel = 2 sessions, 6 postes, soit une création puis
deux rattachements par session :

| `--session` | `--date` | `--poste` | `--motif` |
|---|---|---|---|
| `session_2026-06-18` | `2026-06-18` | `webcam_avant_pause` | `captures_1_images_WIN_20260618_*` |
| `session_2026-06-18` | — | `webcam_apres_pause` | `captures_1.5_images_WIN_20260618_*` |
| `session_2026-06-18` | — | `telephone_hd` | `20260618_*` |
| `session_2026-07-14` | `2026-07-14` | `ali` | `20260714_*_ali.jpg` |
| `session_2026-07-14` | — | `hamza` | `20260714_*_hamza.jpg` |
| `session_2026-07-14` | — | `reda` | `IMG_*_reda.jpg` |

Socle commun : images importées directement `annotee`, labels YOLO convertis
en annotations `humain`/`validee` attribuées au compte dédié **inactif**
`import_historique` (les vrais auteurs par fichier sont inconnus — on ne les
invente pas) ; labels validés contre `data.yaml` (classe hors référentiel =
rejet), absence de retour à la ligne final tolérée ; les fichiers sources ne
sont **jamais** renommés (une collision de nom dans la session cible est
désambiguïsée à l'export : `poste__nom`) ; tout-ou-rien par passe ; relance =
doublons ignorés, rien en double.

## Tests

```bash
cd webapp
.venv/bin/python -m pytest tests -q
```

Les tests exigent un PostgreSQL réel. Résolution automatique :
`TEST_DATABASE_URL` si définie, sinon les binaires portables de
`webapp/.pg16` (cluster jetable créé/détruit par la session pytest).
Le test d'acceptation `test_aller_retour.py` copie deux fois le dataset
réel (stockage de test + export) : si `/tmp` est un tmpfs trop petit
(~10 Go requis), le lancer avec `TMPDIR` sur le disque —
`TMPDIR=$HOME/.cache/compost_tmp .venv/bin/python -m pytest tests -q`.
La suite couvre les garanties du schéma (SQLSTATE exacts), l'égalité
comportementale trigger CW002 ↔ `models.IMAGE_STATUS_TRANSITIONS` (paire par
paire), la non-dérive modèles ↔ base migrée, l'authentification, les rôles et
la CLI.

## Déploiement

```bash
docker compose up -d db   # PostgreSQL 16, identifiants exigés via .env
```
