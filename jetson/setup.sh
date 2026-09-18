#!/usr/bin/env bash
# =============================================================================
# Mise en service « en un clic » d'une Jetson Compost.
#
#     ./setup.sh
#
# Le script vérifie que Docker est installé, pose trois questions (adresse du
# serveur, identifiant de la carte, jeton de sécurité), écrit le fichier .env,
# puis construit et lance le kiosque avec « docker compose up -d --build ».
# Il peut être relancé sans risque : les valeurs déjà en place sont proposées
# par défaut. Détails et dépannage : PROCEDURE_DEPLOIEMENT_JETSON.md
# =============================================================================
set -euo pipefail

# Toujours travailler dans le dossier du script (celui de docker-compose.yml),
# quel que soit l'endroit d'où il est lancé
cd "$(dirname "$(readlink -f "$0")")"

# --- Affichage : couleurs seulement si le terminal les gère
if [ -t 1 ] && command -v tput >/dev/null 2>&1 && [ "$(tput colors 2>/dev/null || echo 0)" -ge 8 ]; then
    GRAS=$(tput bold); VERT=$(tput setaf 2); JAUNE=$(tput setaf 3); ROUGE=$(tput setaf 1); RESET=$(tput sgr0)
else
    GRAS=""; VERT=""; JAUNE=""; ROUGE=""; RESET=""
fi
titre()  { printf '\n%s== %s ==%s\n' "$GRAS" "$*" "$RESET"; }
ok()     { printf '%s  [OK] %s%s\n' "$VERT" "$*" "$RESET"; }
avert()  { printf '%s  [!] %s%s\n' "$JAUNE" "$*" "$RESET"; }
erreur() { printf '%s  [ERREUR] %s%s\n' "$ROUGE" "$*" "$RESET" >&2; }

trap 'erreur "Le script a été interrompu par une erreur (détails ci-dessus). Corrigez puis relancez ./setup.sh ; voir PROCEDURE_DEPLOIEMENT_JETSON.md, section Dépannage."' ERR

# Question avec valeur par défaut : demander VARIABLE "question" "défaut"
demander() {
    local __var=$1 question=$2 defaut=$3 saisie
    if [ -n "$defaut" ]; then
        read -r -p "  $question [$defaut] : " saisie
    else
        read -r -p "  $question : " saisie
    fi
    printf -v "$__var" '%s' "${saisie:-$defaut}"
}

# Rogne les espaces (et retours à la ligne) de début et de fin d'une saisie collée
rogner() {
    local s=$1
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    printf '%s' "$s"
}

# Valeur d'une clé du .env existant (sans exécuter le fichier), guillemets éventuels retirés
lire_env() {
    local valeur
    valeur=$(grep -E "^$1=" .env 2>/dev/null | tail -n 1 | cut -d= -f2- || true)
    valeur=${valeur#\"}; valeur=${valeur%\"}; valeur=${valeur#\'}; valeur=${valeur%\'}
    printf '%s' "$valeur"
}

# =============================================================================
printf '\n%s' "$GRAS"
cat <<'EOF'
  =====================================================================
   Kiosque Compost - Mise en service d'une carte Jetson
  =====================================================================
EOF
printf '%s' "$RESET"
cat <<'EOF'

  Bienvenue ! Ce programme prépare cette carte en 4 étapes :
    1. vérification de Docker,
    2. trois questions (adresse du serveur, identifiant de la carte, jeton),
    3. contrôle des poids du modèle,
    4. construction et lancement du kiosque (10 à 20 min la première fois).

  Munissez-vous des informations remises par l'administrateur :
  l'identifiant de cette carte (JETSON_ID) et le jeton de sécurité (SYNC_TOKEN).
  Appuyez sur Ctrl+C à tout moment pour abandonner.
EOF

[ -f docker-compose.yml ] || { erreur "docker-compose.yml introuvable : lancez ce script depuis le dossier jetson/ du projet."; exit 1; }
if [ ! -f /etc/nv_tegra_release ]; then
    avert "Cette machine ne semble pas être une carte NVIDIA Jetson : le kiosque est prévu pour JetPack 7.2 ou plus récent."
fi

# =============================================================================
titre "Étape 1/4 - Vérification de Docker"

if ! command -v docker >/dev/null 2>&1; then
    erreur "Docker n'est pas installé sur cette carte."
    cat <<'EOF'

  Installez-le en copiant ces commandes dans le terminal (une seule fois) :

      sudo apt update
      sudo apt install -y git docker.io docker-compose-v2 nvidia-container x11-xserver-utils
      sudo systemctl enable --now docker
      sudo usermod -aG docker $USER

  Fermez ensuite la session (déconnexion), rouvrez-la, puis relancez ./setup.sh
EOF
    exit 1
fi
ok "Docker installé : $(docker --version)"

if ! docker compose version >/dev/null 2>&1; then
    erreur "Le module « docker compose » est absent."
    cat <<'EOF'

  Installez-le puis relancez ./setup.sh :

      sudo apt update && sudo apt install -y docker-compose-v2
EOF
    exit 1
fi
ok "docker compose disponible : $(docker compose version 2>/dev/null | head -n 1)"

if ! docker info >/dev/null 2>&1; then
    erreur "Docker est installé mais inaccessible avec votre compte (ou le service est arrêté)."
    cat <<'EOF'

  Exécutez ces commandes, fermez la session, rouvrez-la, puis relancez ./setup.sh :

      sudo systemctl enable --now docker
      sudo usermod -aG docker $USER
EOF
    exit 1
fi
ok "Service Docker accessible"

if ! docker info --format '{{range $nom, $rt := .Runtimes}}{{$nom}} {{end}}' 2>/dev/null | grep -qw nvidia; then
    erreur "Le runtime GPU « nvidia » n'est pas déclaré à Docker (indispensable au kiosque)."
    cat <<'EOF'

  Déclarez-le puis relancez ./setup.sh :

      sudo nvidia-ctk runtime configure --runtime=docker
      sudo systemctl restart docker

  (si la commande nvidia-ctk n'existe pas :  sudo apt install -y nvidia-container)
EOF
    exit 1
fi
ok "Runtime GPU nvidia déclaré à Docker"

# =============================================================================
titre "Étape 2/4 - Configuration de la carte"

if [ -d .env ]; then
    erreur ".env est un dossier (créé par Docker lors d'un lancement sans configuration)."
    echo "  Supprimez-le puis relancez :   sudo rm -r .env && ./setup.sh"
    exit 1
fi

# Valeurs par défaut : celles du .env existant (relance du script), sinon celles du projet
API_URL_DEFAUT="https://compost-dns.duckdns.org"
JETSON_ID_DEFAUT=""
SYNC_TOKEN_DEFAUT=""
if [ -f .env ]; then
    avert "Un fichier .env existe déjà : ses valeurs sont proposées par défaut (Entrée pour les conserver)."
    API_URL_DEFAUT=$(lire_env API_URL); API_URL_DEFAUT=${API_URL_DEFAUT:-https://compost-dns.duckdns.org}
    JETSON_ID_DEFAUT=$(lire_env JETSON_ID)
    SYNC_TOKEN_DEFAUT=$(lire_env SYNC_TOKEN)
    # Le modèle jetson-<site>-<lettre> de .env.example n'est pas une valeur
    case "$JETSON_ID_DEFAUT" in *'<'*) JETSON_ID_DEFAUT="" ;; esac
    case "$SYNC_TOKEN_DEFAUT" in REMPLACER_PAR_LE_JETON_FOURNI) SYNC_TOKEN_DEFAUT="" ;; esac
fi

# --- Adresse du serveur
echo
echo "  1) Adresse du serveur (VPS). Ne la changez que sur consigne de l'administrateur."
regex_url='^https?://[A-Za-z0-9._~:/?&=%+-]+$'
while :; do
    demander API_URL "Adresse du serveur (API_URL)" "$API_URL_DEFAUT"
    API_URL=$(rogner "$API_URL")
    API_URL=${API_URL%/}   # sans « / » final
    [[ "$API_URL" =~ $regex_url ]] && break
    avert "Adresse invalide : elle doit commencer par https:// (ou http://), sans espace. Exemple : https://compost-dns.duckdns.org"
done

# --- Identifiant de la carte
echo
echo "  2) Identifiant de cette carte, attribué par l'administrateur (format jetson-<site>-<lettre>, ex. jetson-tas-a)."
echo "     Il doit être UNIQUE : jamais le même identifiant sur deux cartes."
while :; do
    demander JETSON_ID "Identifiant de la carte (JETSON_ID)" "$JETSON_ID_DEFAUT"
    # Même forme canonique que le serveur : minuscules, « : » -> « - » (une adresse MAC est acceptée)
    JETSON_ID=$(rogner "$JETSON_ID" | tr 'A-Z:' 'a-z-')
    [[ "$JETSON_ID" =~ ^[a-z0-9_.-]{1,100}$ ]] && break
    avert "Identifiant vide ou invalide : lettres, chiffres, « - », « _ » et « . » uniquement (ex. jetson-tas-a)."
done

# --- Jeton de sécurité
echo
echo "  3) Jeton de sécurité remis par l'administrateur. La saisie est masquée : collez-le puis appuyez sur Entrée."
while :; do
    if [ -n "$SYNC_TOKEN_DEFAUT" ]; then
        read -r -s -p "  Jeton de sécurité (SYNC_TOKEN) [Entrée = conserver l'actuel, ${#SYNC_TOKEN_DEFAUT} caractères] : " saisie
    else
        read -r -s -p "  Jeton de sécurité (SYNC_TOKEN) : " saisie
    fi
    echo
    SYNC_TOKEN=$(rogner "${saisie:-$SYNC_TOKEN_DEFAUT}")
    # Pas d'espace, de guillemet, de « # » ni de « $ » : le .env est lu tel quel par l'application et par docker compose
    [[ "$SYNC_TOKEN" =~ ^[A-Za-z0-9._~+/=:-]+$ ]] && break
    avert "Jeton vide ou contenant des caractères inattendus (espace, guillemet...). Recopiez-le exactement."
done

# --- Récapitulatif et confirmation
echo
echo "  Récapitulatif :"
echo "    Serveur (API_URL)   : $API_URL"
echo "    Carte (JETSON_ID)   : $JETSON_ID"
echo "    Jeton (SYNC_TOKEN)  : ${SYNC_TOKEN:0:4}... (${#SYNC_TOKEN} caractères)"
read -r -p "  Enregistrer ces valeurs dans .env ? [O/n] : " reponse
case "${reponse:-o}" in
    [oOyY]*) ;;
    *) echo "  Abandon : rien n'a été modifié."; exit 0 ;;
esac

# Le fichier contient un secret : lisible par son propriétaire uniquement
umask_initial=$(umask)
umask 077
cat > .env <<EOF
# Généré par setup.sh le $(date '+%d/%m/%Y à %H:%M'). Contient un secret : ne jamais le versionner
# ni le copier sur une autre carte (JETSON_ID doit rester unique). Relancer ./setup.sh pour modifier.
SYNC_TOKEN=$SYNC_TOKEN
API_URL=$API_URL
JETSON_ID=$JETSON_ID
EOF
chmod 600 .env
umask "$umask_initial"
ok "Fichier .env écrit"

# =============================================================================
titre "Étape 3/4 - Poids du modèle"

mkdir -p weights a_annoter
if [ -s weights/best.engine ]; then
    ok "weights/best.engine présent (version en place : $(cat weights/.current_version 2>/dev/null || echo aucune))"
elif [ -s weights/best.pt ]; then
    avert "weights/best.engine absent, mais weights/best.pt est là : il peut être compilé au premier démarrage."
    echo "     La compilation TensorRT dure 10 à 20 minutes, écran noir pendant ce temps (best.pt est supprimé ensuite)."
    read -r -p "  Compiler weights/best.pt au premier démarrage ? [O/n] : " reponse
    case "${reponse:-o}" in
        [oOyY]*)
            # Même mécanisme que la mise à jour automatique : l'entrypoint du conteneur compile
            # weights/new_best.pt en best.engine avant de lancer le kiosque
            cp weights/best.pt weights/new_best.pt
            [ -f weights/data.yaml ] && cp weights/data.yaml weights/new_data.yaml
            ok "Compilation programmée au premier démarrage"
            ;;
        *)
            avert "Le kiosque ne démarrera pas sans weights/best.engine."
            ;;
    esac
else
    avert "Aucun poids trouvé dans weights/ : le kiosque ne peut pas démarrer sans weights/best.engine."
    cat <<'EOF'
     Copiez le fichier best.engine remis par l'administrateur dans le dossier weights/ (ou best.pt,
     compilé sur la carte), par exemple depuis un PC du même réseau :

         scp best.engine <utilisateur>@<adresse-ip-de-la-jetson>:<ce dossier>/weights/

     Le fichier .env est conservé : relancez ./setup.sh ensuite (Entrée pour garder les réponses).
EOF
    read -r -p "  Continuer quand même (le kiosque redémarrera en boucle jusqu'à l'arrivée du fichier) ? [o/N] : " reponse
    case "${reponse:-n}" in
        [oOyY]*) ;;
        *) echo "  Arrêt ici. Relancez ./setup.sh une fois les poids copiés."; exit 0 ;;
    esac
fi

# =============================================================================
titre "Étape 4/4 - Construction et lancement du kiosque"

# Le conteneur affiche sur l'écran de la Jetson : autoriser l'accès au serveur graphique
if [ -n "${DISPLAY:-}" ] && command -v xhost >/dev/null 2>&1; then
    if xhost +local: >/dev/null 2>&1; then
        ok "Affichage autorisé pour le conteneur (xhost +local:)"
    else
        avert "« xhost +local: » a échoué : exécutez-le depuis un terminal de la session graphique de la Jetson."
    fi
else
    avert "Pas de session graphique détectée (DISPLAY vide) : l'écran restera noir tant que « xhost +local: »"
    echo "     n'a pas été exécuté dans la session graphique (le démarrage automatique ci-dessous s'en charge à chaque ouverture de session)."
fi

echo
echo "  Construction de l'image puis lancement. Première fois : environ 4 Go téléchargés, 10 à 20 minutes."
echo "  Ne fermez pas ce terminal et n'éteignez pas la carte pendant ce temps."
echo
docker compose up -d --build
echo
ok "Kiosque lancé (conteneur compost-kiosk)"

# =============================================================================
titre "Démarrage automatique"

read -r -p "  Lancer le kiosque automatiquement à chaque démarrage de la carte ? [O/n] : " reponse
case "${reponse:-o}" in
    [oOyY]*)
        # Entrée de session GNOME : autorise l'affichage (xhost) puis (re)lance le conteneur,
        # avec le chemin réel de ce dossier à la place du ~/Compost_Waste_Yolo/jetson du modèle
        mkdir -p ~/.config/autostart
        sed "s|cd ~/Compost_Waste_Yolo/jetson|cd '$PWD'|" autostart/compost-kiosk.desktop > ~/.config/autostart/compost-kiosk.desktop
        ok "Entrée installée : ~/.config/autostart/compost-kiosk.desktop"
        echo "     Activez aussi la connexion automatique de la session : Paramètres -> Utilisateurs -> « Connexion automatique »."
        ;;
    *)
        echo "  Démarrage automatique non installé (voir PROCEDURE_DEPLOIEMENT_JETSON.md, étape 6, pour le faire plus tard)."
        ;;
esac

# =============================================================================
titre "Terminé"
cat <<EOF
  Le kiosque démarre : le flux de la caméra apparaît en plein écran d'ici 1 à 2 minutes
  (bordure verte, rouge lors d'une détection).

  Commandes utiles, à lancer depuis ce dossier ($PWD) :
      docker compose logs -f      journal en direct (Ctrl+C pour quitter, le kiosque continue)
      docker compose restart      redémarrer le kiosque
      docker compose down         arrêter le kiosque

  À FAIRE PAR L'ADMINISTRATEUR, sur le serveur : déclarer la carte « $JETSON_ID » dans l'écran « Matériel ».
  Tant que ce n'est pas fait, la carte affiche « HORS LIGNE » et garde ses captures localement.

  Si un modèle est publié sur le serveur, la carte le télécharge à son premier contact puis compile son
  moteur : écran noir pendant environ 10 minutes, c'est normal, ne redémarrez pas la carte.
EOF
