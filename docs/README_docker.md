# Lancer le projet avec Docker

Docker permet de lancer les trois interfaces **sans rien installer d'autre**
(pas de Python, pas de venv, pas de dépendances) : une seule commande, sur
Windows, Mac ou Linux.

## 1. Installer Docker (une fois)

- **Windows / Mac** : installer [Docker Desktop](https://www.docker.com/products/docker-desktop/).
  Sous Windows, garder l'option par défaut « Use WSL 2 based engine ».
- **Linux** : `sudo apt install docker.io docker-compose-v2` (ou suivre
  [docs.docker.com](https://docs.docker.com/engine/install/)), puis
  `sudo usermod -aG docker $USER` et rouvrir la session.

Vérifier : `docker --version` doit répondre.

## 2. Lancer

Depuis la racine du dépôt :

```bash
docker compose up --build
```

La première fois, la construction de l'image prend plusieurs minutes
(téléchargement des dépendances). Ensuite :

| Interface  | Adresse                  |
|------------|--------------------------|
| Annotation | http://localhost:8501    |
| Production | http://localhost:8502    |
| Mobile     | http://«IP-du-PC»:8000 (depuis le téléphone, même WiFi) |

Commandes utiles :

```bash
docker compose up -d              # lancer en arrière-plan
docker compose up annotation      # une seule interface
docker compose logs -f production # suivre les journaux d'un service
docker compose down               # tout arrêter
docker compose up --build         # reconstruire après une modification du code
```

## 3. Où sont les données ?

**Sur ta machine, pas dans les conteneurs.** Le `docker-compose.yml` monte les
dossiers du dépôt (`dataset_recolte/`, `weights/`, `models/`, `runs/`,
`data/`, `configs/`, `exports/`) à l'intérieur des conteneurs : tout ce qui
est annoté, capturé ou produit atterrit dans ces dossiers et **survit** à
l'arrêt ou à la reconstruction des conteneurs. Supprimer un conteneur ne
supprime jamais une donnée.

Conséquence pratique : déployer un nouveau modèle = remplacer
`weights/best.pt` sur la machine (voir `models/README.md`), puis recharger la
page — aucun rebuild d'image.

## 4. Limites à connaître

- **Webcam USB : non.** Un conteneur ne voit pas la webcam de l'hôte (surtout
  sous Windows/Mac). Pour l'onglet Production, utiliser le mode **URL** avec
  une caméra IP — par exemple un téléphone avec l'application « IP Webcam »
  qui diffuse un flux `http://…` sur le WiFi. Pour utiliser une webcam USB,
  lancer l'interface hors Docker (voir `tuto_installation.md`).
- **Entraînement : pas dans Docker.** L'image embarque torch **CPU**
  (l'image resterait sinon ~4 Go plus grosse) : l'inférence est fluide, mais
  les réentraînements se font sur Colab (voir `notebooks/`) ou sur une machine
  avec GPU hors conteneur. L'onglet Réentraîner fonctionnerait, mais lentement.
- **Port déjà pris** (`port is already allocated`) : une interface tourne déjà
  hors Docker sur le même port — l'arrêter, ou changer le port côté gauche du
  mapping dans `docker-compose.yml` (ex. `"8601:8501"`).
