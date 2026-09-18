# Compost Waste Yolo — détection des indésirables dans le compost

Ce projet surveille les caissons de compost collectif avec une caméra et une intelligence
artificielle : quand un objet non compostable (plastique, métal, verre…) est déposé, l'écran du
caisson passe au rouge et une photo est enregistrée. Ces photos servent ensuite à améliorer l'IA,
en continu, depuis une interface web.

Ce document est le **point d'entrée** du projet. Il explique comment le système est organisé,
comment lancer le serveur, comment mettre en service une nouvelle carte, et renvoie vers les
documentations détaillées.

---

## 1. Comment ça marche

Le système repose sur deux types de machines :

```
        CAISSONS DE COMPOST (sur le terrain)                 SERVEUR CENTRAL (VPS, hébergé)
 ┌────────────────────────────────────────────┐        ┌──────────────────────────────────────┐
 │  Jetson « jetson-tas-a »                   │        │                                      │
 │   caméra + écran + IA embarquée            │  photos│   Base de données (PostgreSQL)       │
 │   ─ détecte en direct (bordure verte/rouge)│ ─────▶ │   Stockage des photos                │
 │   ─ enregistre les photos en local         │        │   Interface web : annotation,        │
 │   ─ les envoie dès que le réseau le permet │ ◀───── │     comptes, matériel, modèles IA    │
 └────────────────────────────────────────────┘ modèle │                                      │
 ┌────────────────────────────────────────────┐   IA   │   Publication des nouveaux modèles   │
 │  Jetson « jetson-tas-b »   (idem)          │ ◀────▶ │                                      │
 └────────────────────────────────────────────┘        └──────────────────────────────────────┘
```

- **Le VPS** (un serveur loué chez un hébergeur) gère la **base de données** et **l'interface
  web**. C'est là que les photos arrivent, que l'on annote les images, gère les comptes, déclare
  les cartes et publie les nouvelles versions de l'IA. Il est joignable en HTTPS à l'adresse
  configurée sur les cartes (aujourd'hui `https://compost-dns.duckdns.org`).
- **Les Jetsons** (un mini-ordinateur NVIDIA par caisson) sont des **clients autonomes** : chaque
  carte fait tourner l'IA localement et fonctionne même sans réseau. Elle garde ses photos sur sa
  carte mémoire (au plus 2000 : au-delà, la plus ancienne est effacée pour faire place à la
  nouvelle) et les envoie au serveur dès que la connexion revient. Une fois par heure, elle
  vérifie si une nouvelle version de l'IA a été publiée et, si oui, l'installe toute seule.

Chaque carte est identifiée par un **`JETSON_ID` unique** (par exemple `jetson-tas-a`) : c'est ce
qui permet de savoir de quel caisson vient chaque photo. Toutes les cartes partagent un **jeton de
sécurité** (`SYNC_TOKEN`) qui les autorise à parler au serveur.

---

## 2. Ce que contient ce dépôt

| Dossier / fichier | Rôle | Documentation |
|---|---|---|
| `webapp/` | **Serveur (VPS)** : API, base de données, interface web d'annotation, worker de pré-annotation | [`webapp/README.md`](webapp/README.md) |
| `jetson/` | **Carte embarquée** : kiosque plein écran, synchronisation, mise à jour automatique du modèle, script `setup.sh` | [`jetson/PROCEDURE_DEPLOIEMENT_JETSON.md`](jetson/PROCEDURE_DEPLOIEMENT_JETSON.md) |
| `compost-yolo/` | **Entraînement de l'IA** : préparation du dataset, entraînement, évaluation, export | [`compost-yolo/README.md`](compost-yolo/README.md) |
| `app.py`, `mobile/`, `bbox_editor/` | Outils d'annotation historiques (PC et mobile), antérieurs à l'interface web | [`tuto_installation.md`](tuto_installation.md), [`mobile/README.md`](mobile/README.md) |
| `docs/` | Contexte du projet, captures d'écran | [`docs/CONTEXTE_PROJET.md`](docs/CONTEXTE_PROJET.md) |

---

## 3. Lancer le serveur (VPS)

À faire une fois, sur le VPS (Linux). Prérequis : `git`, Docker, Python 3.12 et Node.js (pour
construire l'interface).

```bash
# 1. Récupérer le projet
git clone https://github.com/TSResearch-hub/Compost_Waste_Yolo.git
cd Compost_Waste_Yolo/webapp

# 2. Configurer (le fichier .env n'est jamais partagé ni versionné)
cp .env.example .env
nano .env
```

Dans `.env`, renseignez au minimum :

| Variable | À mettre |
|---|---|
| `POSTGRES_PASSWORD` | un mot de passe pour la base de données |
| `DATABASE_URL` | `postgresql+psycopg://compost:<POSTGRES_PASSWORD>@localhost:5432/compost_annotation` |
| `STORAGE_ROOT` | le dossier où stocker les photos (disque de bonne taille) |
| `DATA_YAML_PATH` | le chemin absolu de `compost-yolo/configs/data.yaml` (liste des classes) |
| `SYNC_TOKEN` | le jeton de sécurité des cartes, généré avec `python3 -c "import secrets; print(secrets.token_urlsafe(32))"` — **c'est ce jeton qu'il faut donner à l'installateur de chaque Jetson** |
| `WEIGHTS_PATH` | le fichier `.pt` du modèle courant (utilisé par le worker de pré-annotation) |

```bash
# 3. Base de données (PostgreSQL 16 dans Docker, redémarre seule)
docker compose up -d db

# 4. Application et premier compte administrateur
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/alembic upgrade head
.venv/bin/python -m app.cli create-admin --username admin      # le mot de passe est demandé

# 5. Interface web (à refaire après chaque mise à jour du code)
cd frontend && npm install && npm run build && cd ..

# 6. Lancer le serveur — l'interface est alors sur http://<adresse-du-vps>:8000
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Le **worker de pré-annotation** (l'IA propose des boîtes que l'annotateur valide) est un second
processus, facultatif mais très utile :

```bash
python3 -m venv .venv-worker
.venv-worker/bin/pip install -r requirements-worker.txt
.venv-worker/bin/python -m app.worker
```

En production, ces deux processus doivent être lancés comme services au démarrage (systemd) et le
serveur exposé en HTTPS (reverse proxy, certificat) à l'adresse `API_URL` connue des cartes : cette
partie est propre au VPS et n'est pas dans ce dépôt. **À sauvegarder régulièrement** : la base de
données (`docker compose exec -T db pg_dump -U compost compost_annotation > sauvegarde.sql`), le
dossier `STORAGE_ROOT` et le fichier `.env`.

### Les écrans de l'interface (compte administrateur)

| Écran | Sert à |
|---|---|
| **Lots** | annoter les images (lots, canvas, relecture) — un guide d'annotation est intégré à l'interface |
| **Comptes** | créer les comptes des annotateurs et administrateurs |
| **Matériel** | déclarer chaque carte Jetson (identifiant + nom) — une carte non déclarée est refusée par le serveur |
| **Modèles IA** | publier une nouvelle version du modèle, déployée automatiquement sur toutes les cartes |
| **Dataset** | télécharger sur son PC le dataset d'entraînement (ZIP : images, labels YOLO, `data.yaml`) et déposer un ZIP d'images brutes à annoter |
| **Technique** | importer des dossiers d'images déjà sur le serveur, exporter le dataset dans un dossier du serveur, surveiller la pré-annotation |

Tous les détails (garanties de la base, API, tests, import de l'historique) sont dans
[`webapp/README.md`](webapp/README.md).

---

## 4. Mettre en service une nouvelle Jetson

### Avant de commencer (administrateur)

1. Choisir un **identifiant unique** pour la carte, au format `jetson-<site>-<lettre>` (exemple :
   `jetson-tas-a`). Ne jamais réutiliser l'identifiant d'une autre carte ; le noter sur une
   étiquette collée sur le caisson.
2. Le **déclarer sur le serveur** : écran **Matériel** → « Nouvelle carte Jetson » (identifiant, nom).
3. Remettre à la personne qui installe : cet identifiant, le **jeton `SYNC_TOKEN`** (celui du `.env`
   du serveur) et le fichier de poids du modèle **`best.engine`** (ou `best.pt`, compilé sur place).

### Sur la carte (installateur)

Matériel attendu : carte Jetson flashée avec **JetPack 7.2 ou plus récent**, écran, caméra USB,
encodeur 3 boutons, accès Internet, et une session graphique ouverte.

```bash
# 1. Récupérer le projet (si git manque : sudo apt install -y git)
git clone https://github.com/TSResearch-hub/Compost_Waste_Yolo.git ~/Compost_Waste_Yolo
cd ~/Compost_Waste_Yolo/jetson

# 2. Déposer les poids du modèle dans weights/ (par clé USB, ou depuis un PC du même réseau :
#    scp best.engine <utilisateur>@<adresse-ip-de-la-jetson>:~/Compost_Waste_Yolo/jetson/weights/)

# 3. Lancer l'installation guidée
./setup.sh
```

Le script `setup.sh` fait tout le reste, en posant trois questions :

1. il vérifie que **Docker** est installé et prêt (sinon il affiche exactement quoi installer) ;
2. il demande **l'adresse du serveur** (valeur par défaut proposée : Entrée pour l'accepter),
   **l'identifiant de la carte** et **le jeton de sécurité** (saisie masquée : coller, puis Entrée) ;
3. il écrit le fichier `.env` de la carte, vérifie les poids du modèle, puis construit et lance le
   kiosque (`docker compose up -d --build` — **10 à 20 minutes** la première fois, environ 4 Go
   téléchargés) ;
4. il propose d'installer le **démarrage automatique** : la carte relance le kiosque toute seule à
   chaque allumage (activer aussi la « Connexion automatique » dans Paramètres → Utilisateurs).

Le script peut être relancé sans risque (par exemple pour changer le jeton) : il propose les valeurs
déjà en place. Pour vérifier que tout va bien : le flux caméra s'affiche en plein écran avec une
bordure verte, et `docker compose logs -f` montre des lignes `Sync : … envoyée(s) au VPS`. Si un
modèle est publié sur le serveur, la carte le télécharge à son premier contact et compile son
moteur : **écran noir pendant environ 10 minutes, c'est normal**.

Deux points importants, détaillés dans la
[procédure complète](jetson/PROCEDURE_DEPLOIEMENT_JETSON.md) (matériel, dépannage, exploitation) :

- un bandeau **« HORS LIGNE »** permanent signifie que le serveur refuse la carte : identifiant non
  déclaré dans l'écran **Matériel**, jeton erroné, ou pas d'Internet ;
- un **nouveau caisson change les images** (éclairage, angle, fond) : prévoir une campagne de
  captures manuelles (bouton 0, 200 à 500 photos) puis un ré-entraînement, sans quoi l'IA peut
  rater des indésirables ou déclencher de fausses alertes.

---

## 5. Faire vivre l'IA (boucle d'amélioration)

1. **Les cartes envoient leurs photos** au serveur : automatiquement à chaque alerte, ou à la
   demande avec le bouton 0. Elles apparaissent dans des sessions `<jetson-id>_<date>`. Des
   photos prises autrement (appareil photo, téléphone) se déposent en ZIP depuis l'écran
   **Dataset** → « Importer des images brutes ».
2. **Annoter** dans l'interface web (écran **Lots**) : l'IA propose des boîtes, l'annotateur les
   valide, corrige ou rejette.
3. **Exporter** le dataset au format YOLO : écran **Dataset** → « Télécharger le dataset »
   (un ZIP sur votre PC), ou écran **Technique** → Export (dans un dossier du serveur).
4. **Ré-entraîner** le modèle avec les outils de [`compost-yolo/`](compost-yolo/README.md) (PC avec
   GPU ou Google Colab) : on obtient un nouveau `best.pt` et son `data.yaml`.
5. **Publier** cette version dans l'écran **Modèles IA** (nom de version, `.pt`, `data.yaml`,
   « Déployer sur la flotte »).
6. **Chaque carte se met à jour seule** dans l'heure : téléchargement, compilation du moteur
   (≈ 10 minutes d'écran noir), redémarrage du kiosque avec le nouveau modèle. Rien à faire sur
   les cartes.

---

## 6. Documentation détaillée

| Document | Contenu |
|---|---|
| [`webapp/README.md`](webapp/README.md) | Serveur : installation, migrations, comptes, import/export, flotte Jetson, distribution des modèles, worker, tests |
| [`jetson/PROCEDURE_DEPLOIEMENT_JETSON.md`](jetson/PROCEDURE_DEPLOIEMENT_JETSON.md) | Carte : procédure pas à pas, exploitation courante, **tableau de dépannage**, campagne de captures |
| [`jetson/setup.sh`](jetson/setup.sh) | Carte : installation guidée (ce que fait chaque étape est commenté dans le script) |
| [`compost-yolo/README.md`](compost-yolo/README.md) | IA : préparer le dataset, entraîner, évaluer, exporter ; [`README_import.md`](compost-yolo/README_import.md) (datasets externes), [`README_annotation.md`](compost-yolo/README_annotation.md) |
| [`tuto_installation.md`](tuto_installation.md), [`mobile/README.md`](mobile/README.md) | Outils d'annotation historiques sur PC (Streamlit) et mobile |
| [`docs/CONTEXTE_PROJET.md`](docs/CONTEXTE_PROJET.md) | Contexte, objectifs, genèse du projet et démonstration vidéo |

---

## 7. En cas de problème

| Symptôme | Où regarder |
|---|---|
| Une carte affiche un écran noir, « HORS LIGNE », ou redémarre en boucle | Sur la carte : `cd ~/Compost_Waste_Yolo/jetson && docker compose logs --tail 50`, puis le [tableau de dépannage](jetson/PROCEDURE_DEPLOIEMENT_JETSON.md#8-dépannage) |
| Le serveur ne répond pas | Sur le VPS : `curl http://localhost:8000/api/health`, `docker compose ps` (base de données), journal du service `uvicorn` |
| Les photos d'une carte n'arrivent pas | Écran **Matériel** : la carte est-elle déclarée et « en service » ? Le `JETSON_ID` saisi sur la carte est-il exactement le même ? |
| Les cartes ne récupèrent pas un modèle publié | Attendre jusqu'à une heure ; sur la carte, `cat weights/.current_version` puis `docker compose logs -f` (lignes `Modèle : …`) |
