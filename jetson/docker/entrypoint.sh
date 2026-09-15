#!/bin/bash
# =============================================================================
# Point d'entrée du conteneur kiosque Compost.
#   - Pour la commande du kiosque (app_kiosk.py) : contrôle la configuration, compile en
#     moteur TensorRT une nouvelle version des poids si l'application en a téléchargé une
#     (weights/new_best.pt), contrôle les poids et les périphériques, puis attend que le
#     serveur X11 de l'hôte soit disponible (au boot, le conteneur démarre souvent avant
#     la session graphique).
#   - Pour toute autre commande (ex. "yolo export ...") : exécution directe.
# =============================================================================
set -euo pipefail
cd /app

log() { echo "[entrypoint] $*" >&2; }

case "$*" in
    *app_kiosk.py*)
        # 1) Configuration .env (montée depuis l'hôte)
        if [ -d /app/.env ]; then
            log "ERREUR : /app/.env est un dossier. Le fichier .env n'existait pas sur l'hôte au premier lancement :"
            log "         supprimez le dossier .env créé par Docker, copiez .env.example en .env, puis relancez."
            exit 1
        fi
        if [ ! -s /app/.env ]; then
            log "ERREUR : /app/.env absent ou vide. Copiez .env.example en .env et renseignez JETSON_ID, SYNC_TOKEN, API_URL."
            exit 1
        fi
        if ! grep -Eq '^JETSON_ID=[A-Za-z0-9._-]+$' /app/.env || grep -Eq '^JETSON_ID=jetson-<' /app/.env; then
            log "ERREUR : JETSON_ID n'est pas renseigné dans .env (identifiant unique obligatoire, ex. jetson-tas-a)."
            exit 1
        fi

        # 2) Nouvelle version des poids téléchargée par l'application (sync_worker de app_kiosk.py :
        #    weights/new_best.pt + weights/new_data.yaml, puis arrêt du programme -> Docker relance le
        #    conteneur et on arrive ici). Un .engine est lié au GPU et à la version de TensorRT : il se
        #    compile sur la carte elle-même, AVANT le lancement du kiosque (environ 10 min, écran noir).
        if [ -f /app/weights/new_best.pt ]; then
            version="$(cat /app/weights/.current_version 2>/dev/null || echo inconnue)"
            log "Nouvelle version détectée ($version), compilation TensorRT en cours (environ 10 min)..."
            # Restes d'une compilation interrompue (le .onnx est un intermédiaire créé par l'export)
            rm -f /app/weights/new_best.engine /app/weights/new_best.onnx
            if python3 -c "from ultralytics import RTDETR; model = RTDETR('weights/new_best.pt'); model.export(format='engine', imgsz=640)" \
               && [ -s /app/weights/new_best.engine ]; then
                mv -f /app/weights/new_best.engine /app/weights/best.engine
                if [ -f /app/weights/new_data.yaml ]; then
                    mv -f /app/weights/new_data.yaml /app/weights/data.yaml
                fi
                # Libération de l'espace : intermédiaires new_* et poids .pt (l'ancien best.pt ne
                # correspond plus au moteur en place ; le VPS conserve la source de chaque version)
                rm -f /app/weights/new_* /app/weights/*.pt
                log "Compilation terminée : weights/best.engine et weights/data.yaml remplacés (version $version)."
            else
                log "ERREUR : la compilation TensorRT de weights/new_best.pt a échoué (détails ci-dessus)."
                log "         L'ancien weights/best.engine est conservé et le kiosque démarre avec."
                log "         Pour réessayer : sudo rm weights/.current_version puis docker compose restart"
                rm -f /app/weights/new_*
            fi
        fi

        # 3) Poids du modèle TensorRT
        if [ ! -s /app/weights/best.engine ]; then
            log "ERREUR : weights/best.engine introuvable. Déposez-le dans le dossier weights/ de l'hôte,"
            log "         ou générez-le : docker compose run --rm kiosk yolo export model=weights/best.pt format=engine imgsz=640"
            exit 1
        fi

        # 4) Périphériques (simples avertissements : l'application affiche sa propre erreur)
        [ -e /dev/video0 ] || log "AVERTISSEMENT : /dev/video0 absent, la caméra n'est pas visible dans le conteneur."
        ls /dev/input/event* >/dev/null 2>&1 || log "AVERTISSEMENT : aucun périphérique dans /dev/input, l'encodeur ne sera pas détecté."

        # 5) Serveur X11 : on patiente jusqu'à 2 minutes (démarrage de la session graphique)
        export DISPLAY="${DISPLAY:-:0}"
        for i in $(seq 1 60); do
            xdpyinfo >/dev/null 2>&1 && break
            [ "$i" -eq 1 ] && log "En attente du serveur X11 (DISPLAY=$DISPLAY)..."
            sleep 2
        done
        if xdpyinfo >/dev/null 2>&1; then
            log "Serveur X11 disponible sur DISPLAY=$DISPLAY"
        else
            log "AVERTISSEMENT : X11 injoignable après 2 min (vérifiez 'xhost +local:' sur l'hôte). Tentative de lancement."
        fi
        ;;
esac

exec "$@"
