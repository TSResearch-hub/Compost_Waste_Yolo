# Procédure de mise en service d'une Jetson « Compost »

Ce document explique comment installer le kiosque de détection Compost sur une **nouvelle carte NVIDIA Jetson**, de façon identique d'une carte à l'autre. L'application tourne dans un conteneur Docker : le même conteneur est utilisé sur toutes les cartes, seuls le fichier `.env` (identité de la carte) et les poids du modèle changent.

Le kiosque affiche le flux de la caméra en plein écran, détecte les indésirables dans le compost grâce au modèle IA, enregistre des captures localement dans `a_annoter/`, puis les envoie au serveur (VPS) en tâche de fond. En cas de coupure réseau, les captures restent sur la carte et repartent dès le retour de la connexion (bandeau rouge « HORS LIGNE - ENREGISTREMENT LOCAL » à l'écran).

---

## 0. Ce dont vous avez besoin

| Élément | Détail |
|---|---|
| Carte Jetson | **Même modèle que la carte de référence**, flashée avec **JetPack 7.2 ou plus récent** (Ubuntu 24.04, Python 3.12). Une carte sous JetPack 5 ou 6 n'est pas compatible avec cette image. |
| Périphériques | Écran (HDMI/DP), caméra USB 1080p (MJPEG), encodeur 3 boutons USB, connexion Internet (HTTPS sortant). |
| Session graphique | Un compte utilisateur avec **ouverture de session automatique** (le kiosque a besoin de l'écran). |
| Fournis par l'administrateur | L'adresse du dépôt Git, le jeton `SYNC_TOKEN`, un **`JETSON_ID` unique attribué à cette carte**, et les fichiers de poids `best.engine` et `best.pt`. |

Vérification rapide de la version JetPack sur la carte :

```bash
cat /etc/nv_tegra_release      # doit indiquer R39 (JetPack 7.2) ou plus récent
python3 --version              # doit indiquer Python 3.12
```

---

## 1. Préparer la carte (une seule fois)

Ouvrez un terminal sur la Jetson (ou en SSH) et installez les outils :

```bash
sudo apt update
sudo apt install -y git docker.io docker-compose-v2 nvidia-container x11-xserver-utils
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
```

Déconnectez-vous puis reconnectez-vous pour que l'ajout au groupe `docker` prenne effet. Vérifiez ensuite que Docker connaît le runtime GPU :

```bash
docker info | grep -i runtimes     # la ligne doit contenir "nvidia"
```

Si `nvidia` n'apparaît pas :

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Si la carte a déjà servi avec l'ancien lancement manuel** (script `launch_kiosk.sh` au démarrage), supprimez cet ancien démarrage automatique : deux instances se disputeraient la caméra.

```bash
ls ~/.config/autostart/            # supprimez l'entrée qui lance launch_kiosk.sh, s'il y en a une
```

---

## 2. Récupérer l'application

```bash
git clone <URL_DU_DEPOT> ~/Compost_Jetson
cd ~/Compost_Jetson
```

Le dossier doit s'appeler `Compost_Jetson` et se trouver dans votre dossier personnel : le démarrage automatique (étape 6) compte dessus.

---

## 3. Configurer le fichier `.env`

```bash
cp .env.example .env
nano .env
chmod 600 .env
```

Renseignez les trois valeurs :

```
SYNC_TOKEN=<jeton fourni par l'administrateur>
API_URL=https://compost-dns.duckdns.org
JETSON_ID=<identifiant attribué à cette carte>
```

### ⚠️ Règle absolue : un `JETSON_ID` unique par carte

- L'identifiant est attribué par l'administrateur, au format `jetson-<site>-<lettre>` (exemple : `jetson-tas-a`).
- **Ne réutilisez jamais l'identifiant d'une autre carte** et **ne copiez jamais le `.env` d'une carte à l'autre**.
- Le serveur classe chaque capture reçue sous ce `JETSON_ID`. Deux cartes avec le même identifiant mélangent leurs images : impossible ensuite de savoir de quel caisson vient une capture, ni de ré-entraîner correctement le modèle pour ce caisson.
- Notez l'identifiant sur une étiquette collée sur le caisson.

Le conteneur refuse de démarrer si `JETSON_ID` n'est pas renseigné.

---

## 4. Déposer les poids du modèle

Les fichiers de poids ne sont pas dans le dépôt Git. Copiez-les dans le dossier `weights/` :

| Fichier | Rôle |
|---|---|
| `weights/best.engine` | **Obligatoire.** Modèle optimisé TensorRT, chargé par l'application. |
| `weights/best.pt` | Facultatif. Modèle source, utile pour régénérer `best.engine` à la main. Il est supprimé lors de la première mise à jour automatique (ci-dessous). |

Depuis un PC sur le même réseau :

```bash
scp best.engine best.pt <utilisateur>@<adresse-ip-de-la-jetson>:~/Compost_Jetson/weights/
```

**Important.** Un fichier `.engine` est lié au modèle de GPU et à la version de TensorRT qui l'ont produit. Si la nouvelle carte est d'un modèle différent, ou si le journal affiche une erreur au chargement du moteur (« deserialize », « version mismatch »), régénérez-le sur la carte à partir de `best.pt` (durée : 10 à 20 minutes) :

```bash
docker compose run --rm kiosk yolo export model=weights/best.pt format=engine imgsz=640
```

Le nouveau `weights/best.engine` remplace l'ancien.

### Mise à jour automatique du modèle

Une fois le kiosque lancé, il n'y a plus de poids à copier à la main. La carte interroge le serveur dès le démarrage puis **une fois par heure** : si l'administrateur a publié une version différente de celle notée dans `weights/.current_version` (« aucune » tant que la carte n'a jamais été mise à jour), la carte télécharge les nouveaux poids, **se relance et compile son propre moteur TensorRT**. Pendant cette compilation (environ 10 minutes), **l'écran reste noir** et le journal affiche « Nouvelle version détectée, compilation TensorRT en cours » : c'est normal, ne redémarrez pas la carte. Le kiosque reprend ensuite seul avec le nouveau modèle.

Conséquence : au premier lancement, la carte télécharge la version publiée même si vous venez de copier un `best.engine` identique. Comptez donc 10 minutes d'écran noir peu après la première mise en service.

---

## 5. Autoriser l'affichage, construire et lancer

Depuis un terminal **ouvert dans la session graphique de la Jetson** :

```bash
cd ~/Compost_Jetson
xhost +local:                 # autorise le conteneur à utiliser l'écran
docker compose build          # première fois : environ 4 Go téléchargés, 10 à 20 minutes
docker compose up -d
docker compose logs -f        # Ctrl+C pour quitter le journal (le kiosque continue)
```

Au premier lancement, le modèle met quelques dizaines de secondes à se charger. Le journal doit afficher, dans l'ordre :

```
[entrypoint] Serveur X11 disponible sur DISPLAY=:0
Version des poids en place : aucune
Encodeur détecté : ...
Chargement du moteur TensorRT...
Interface lancée. ...
Sync : N image(s) envoyée(s) au VPS et supprimée(s) localement
Modèle : version <nom> à jour          (ou : NOUVELLE VERSION DISPONIBLE ... puis redémarrage et compilation, voir étape 4)
```

À l'écran : le flux vidéo plein écran avec une bordure verte (rouge lors d'une détection). Si le bandeau rouge « HORS LIGNE - ENREGISTREMENT LOCAL » reste affiché, voir la section Dépannage.

---

## 6. Démarrage automatique au boot

Docker relance le conteneur automatiquement à chaque démarrage de la carte. Il reste à autoriser l'affichage à chaque ouverture de session :

```bash
mkdir -p ~/.config/autostart
cp ~/Compost_Jetson/autostart/compost-kiosk.desktop ~/.config/autostart/
```

Activez ensuite l'ouverture de session automatique : Paramètres → Utilisateurs → « Connexion automatique ». Redémarrez la carte pour valider : le kiosque doit apparaître seul, sans intervention.

---

## 7. Exploitation courante

| Besoin | Commande (dans `~/Compost_Jetson`) |
|---|---|
| Voir le journal en direct | `docker compose logs -f` |
| Redémarrer le kiosque | `docker compose restart` |
| Arrêter le kiosque | `docker compose down` |
| Relancer après un arrêt | `docker compose up -d` |
| Mettre à jour l'application | `git pull && docker compose build && docker compose up -d` |
| Vérifier l'accès GPU | `docker compose run --rm kiosk python -c "import torch; print(torch.cuda.is_available())"` |
| Voir la version du modèle en place | `cat weights/.current_version` |
| Forcer le re-téléchargement et la recompilation du modèle publié | `sudo rm weights/.current_version && docker compose restart` (écran noir ~10 min) |

Les captures en attente d'envoi sont dans `~/Compost_Jetson/a_annoter/`. Le dossier se vide au fur et à mesure des envois réussis. Les fichiers y appartiennent à `root` (créés par le conteneur) : utilisez `sudo` pour les manipuler.

---

## 8. Dépannage

| Symptôme | Cause probable | Que faire |
|---|---|---|
| Écran noir, journal : « could not connect to display » ou « En attente du serveur X11 » | L'affichage n'est pas autorisé pour le conteneur | Dans la session graphique : `xhost +local:` puis `docker compose up -d --force-recreate`. Vérifiez l'entrée de démarrage automatique (étape 6). |
| Journal : « Aucun joystick détecté » et redémarrages en boucle | Encodeur débranché, ou branché après le démarrage du conteneur | Branchez l'encodeur puis `docker compose restart`. |
| Journal : « Impossible d'ouvrir la caméra » | La caméra n'est pas sur `/dev/video0` | Sur l'hôte : `v4l2-ctl --list-devices`. Adaptez la ligne `/dev/video0` dans `docker-compose.yml`, puis `docker compose up -d`. |
| Erreur au chargement du moteur (« deserialize », « version mismatch ») | `best.engine` produit sur un autre GPU ou une autre version de TensorRT | Régénérez-le (étape 4), ou forcez la mise à jour automatique : `sudo rm weights/.current_version && docker compose restart`. |
| Écran noir, journal : « Nouvelle version détectée, compilation TensorRT en cours » | Mise à jour automatique du modèle en cours | Attendez environ 10 minutes sans redémarrer la carte. |
| Journal : « la compilation TensorRT de weights/new_best.pt a échoué » | Poids publiés incompatibles, ou incident pendant la compilation | Le kiosque continue avec l'ancien modèle. Lisez l'erreur dans `docker compose logs`, prévenez l'administrateur ; pour réessayer : `sudo rm weights/.current_version && docker compose restart`. |
| Bandeau « HORS LIGNE » permanent | `.env` incorrect, jeton invalide, DNS ou Internet indisponible | `docker compose logs --tail 50` (lignes « Sync : … »), vérifiez `.env`, puis `curl -I https://compost-dns.duckdns.org`. Surveillez l'espace disque avec `df -h` : le tampon local grossit tant que l'envoi échoue. |
| « unknown or invalid runtime name: nvidia » | Runtime GPU non déclaré à Docker | `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker` |
| « permission denied » sur `docker` | Utilisateur hors du groupe `docker` | `sudo usermod -aG docker $USER` puis déconnexion/reconnexion. |
| `.env` refusé : « est un dossier » | Le conteneur a été lancé avant la création du `.env` | `sudo rm -r .env`, puis reprenez l'étape 3. |

---

## 9. ⚠️ RAPPEL IMPORTANT : un nouveau caisson change le domaine visuel

Le modèle IA a appris à reconnaître les indésirables **sur les images du caisson de référence** : son éclairage, sa couleur de fond, la hauteur et l'angle de sa caméra, ses reflets. Un nouveau caisson, même construit sur le même plan, produit des images différentes. C'est ce qu'on appelle un **écart de domaine (domain gap)**. Conséquence : sur une nouvelle installation, le modèle peut rater des indésirables ou déclencher de fausses alertes, sans que rien ne soit « cassé ».

**La mise en service d'un nouveau caisson doit donc toujours s'accompagner d'une campagne de captures pour ré-entraîner le modèle :**

1. Dès l'installation, laissez le kiosque fonctionner normalement et vérifiez que la synchronisation marche (pas de bandeau « HORS LIGNE »). Les captures partent sur le VPS sous le `JETSON_ID` de la carte.
2. Réalisez des **captures manuelles (bouton 0)** dans des situations variées : bac vide, compost seul, chaque type d'indésirable, différentes heures et conditions d'éclairage. Visez de l'ordre de **200 à 500 captures** sur les premiers jours.
3. Prévenez l'administrateur : il annote les nouvelles images sur le VPS et ré-entraîne le modèle en y intégrant ce nouveau domaine.
4. L'administrateur publie la nouvelle version sur le serveur (écran « Modèles IA »). Chaque carte connectée la récupère d'elle-même dans l'heure, se relance et compile son moteur (écran noir environ 10 minutes, voir étape 4). Rien à faire sur les cartes.

Tant que ce ré-entraînement n'a pas eu lieu, considérez les détections du nouveau caisson comme indicatives. Et n'oubliez pas : c'est le `JETSON_ID` unique qui permet de relier chaque capture à son caisson.

---

## Liste de contrôle finale

- [ ] JetPack 7.2+ vérifié (`cat /etc/nv_tegra_release`)
- [ ] Docker fonctionnel avec le runtime `nvidia`, utilisateur dans le groupe `docker`
- [ ] Ancien lancement `launch_kiosk.sh` désactivé (si la carte a déjà servi)
- [ ] Dépôt cloné dans `~/Compost_Jetson`
- [ ] `.env` créé avec un **`JETSON_ID` unique**, étiquette collée sur le caisson
- [ ] `weights/best.engine` en place (moteur régénéré si nécessaire ; `best.pt` facultatif)
- [ ] `xhost +local:` puis `docker compose build` et `docker compose up -d` réussis
- [ ] Flux vidéo plein écran visible, encodeur détecté, ligne « Sync : … envoyée(s) » dans le journal
- [ ] Démarrage automatique installé et validé par un redémarrage
- [ ] Campagne de captures planifiée pour le ré-entraînement (section 9)
