import io
import os
import subprocess
import sys
import threading
import time
import zipfile
from datetime import datetime
import cv2
import pygame
import requests
from dotenv import load_dotenv
from ultralytics import RTDETR

# ==========================================
# 1. INITIALISATION DE L'ENVIRONNEMENT
# ==========================================
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts.warning=false"
os.environ["SDL_VIDEODRIVER"] = "dummy"

DOSSIER_ANNOTATION = "a_annoter"
os.makedirs(DOSSIER_ANNOTATION, exist_ok=True)

# Configuration de la synchronisation VPS (fichier .env à côté de ce script)
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
SYNC_TOKEN = os.getenv("SYNC_TOKEN")
API_URL = (os.getenv("API_URL") or "").rstrip("/")
JETSON_ID = os.getenv("JETSON_ID")
SYNC_INTERVALLE = 60            # secondes entre deux tentatives d'envoi
SYNC_MAX_IMAGES_PAR_LOT = 100   # borne la taille du ZIP en mémoire (0 = tout envoyer d'un coup)
SYNC_AGE_MIN = 2.0              # secondes : ignore une image encore en cours d'écriture

# Mise à jour automatique des poids du modèle (voir section 3 et docker/entrypoint.sh)
MODELE_INTERVALLE = 3600        # secondes entre deux vérifications de la version publiée sur le VPS (1 h = 60 cycles)
DOSSIER_POIDS = "weights"
FICHIER_VERSION = os.path.join(DOSSIER_POIDS, ".current_version")   # version_name des poids en place
VERSION_AUCUNE = "aucune"       # contenu par défaut de FICHIER_VERSION : poids déposés à la main, version inconnue
NOUVEAU_PT = os.path.join(DOSSIER_POIDS, "new_best.pt")             # compilé en best.engine par docker/entrypoint.sh
NOUVEAU_YAML = os.path.join(DOSSIER_POIDS, "new_data.yaml")         # devient weights/data.yaml

SYNC_CONFIG_OK = bool(SYNC_TOKEN and API_URL and JETSON_ID)
if not SYNC_CONFIG_OK:
    print("Attention : SYNC_TOKEN, API_URL ou JETSON_ID manquant dans .env, la synchronisation VPS et la mise à jour du modèle sont désactivées.")
# Même jeton pour l'envoi des captures et la distribution des modèles (Authorization: Bearer, attendu par le VPS)
EN_TETES_SYNC = {"Authorization": f"Bearer {SYNC_TOKEN}", "X-Jetson-Id": JETSON_ID} if SYNC_CONFIG_OK else {}


def ecrire_version_locale(version):
    os.makedirs(DOSSIER_POIDS, exist_ok=True)
    with open(FICHIER_VERSION, "w", encoding="utf-8") as f:
        f.write(version + "\n")


def lire_version_locale():
    """Retourne le version_name des poids en place (weights/.current_version).

    Le fichier est créé avec "aucune" s'il n'existe pas (ou s'il est vide) : une carte dont les
    poids ont été déposés à la main télécharge alors la version publiée sur le VPS dès la première
    vérification, et toute la flotte converge vers la même version.
    """
    try:
        with open(FICHIER_VERSION, encoding="utf-8") as f:
            version = f.read().strip()
        if version:
            return version
    except FileNotFoundError:
        pass
    ecrire_version_locale(VERSION_AUCUNE)
    return VERSION_AUCUNE


print(f"Version des poids en place : {lire_version_locale()}")

pygame.init()
pygame.joystick.init()

joystick = None
if pygame.joystick.get_count() > 0:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Encodeur détecté : {joystick.get_name()}")
else:
    print("Attention : Aucun joystick détecté, contrôle clavier uniquement ('q', 'c').")
    exit()

print("Chargement du moteur TensorRT...")
model = RTDETR("weights/best.engine")

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la caméra.")
    exit()

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

nom_fenetre = "Compost - Kiosk Mode"
fenetre_ouverte = False

def ouvrir_fenetre():
    cv2.namedWindow(nom_fenetre, cv2.WINDOW_GUI_NORMAL)
    cv2.setWindowProperty(nom_fenetre, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        subprocess.run(["wmctrl", "-r", nom_fenetre, "-b", "add,fullscreen,above"], check=False)
    except Exception:
        pass
    global fenetre_ouverte
    fenetre_ouverte = True

# Ouverture initiale
ouvrir_fenetre()

# ==========================================
# 2. VARIABLES D'ÉTAT
# ==========================================
running = True
en_veille = False
afficher_ia = True

message_temporaire = ""
fin_message = 0

etat_precedent_boutons = {0: False, 1: False, 2: False}

# Variables pour l'enregistrement automatique des alertes
alerte_precedente = False
temps_derniere_alerte = 0.0

# État de la connexion au VPS (mis à jour par le thread de synchronisation)
reseau_ok = True

# ==========================================
# 3. SYNCHRONISATION VPS ET MISE À JOUR DU MODÈLE (TÂCHE DE FOND)
# ==========================================
def telecharger_fichier(url, destination):
    """Télécharge `url` vers `destination` par morceaux d'1 Mo (stream=True : un .pt pèse des
    dizaines de Mo, rien n'est chargé en RAM). Écrit d'abord dans un fichier temporaire, contrôle
    la taille annoncée par le serveur (Content-Length), puis renomme : le nom final ne désigne
    jamais un fichier tronqué. Retourne la taille écrite en octets."""
    temporaire = destination + ".part"
    taille = 0
    try:
        with requests.get(url, headers=EN_TETES_SYNC, stream=True, timeout=(10, 300)) as reponse:
            reponse.raise_for_status()
            taille_attendue = int(reponse.headers.get("Content-Length") or 0)
            with open(temporaire, "wb") as f:
                for morceau in reponse.iter_content(chunk_size=1024 * 1024):
                    f.write(morceau)
                    taille += len(morceau)
        if taille == 0 or (taille_attendue and taille != taille_attendue):
            raise IOError(f"téléchargement incomplet de {os.path.basename(destination)} ({taille}/{taille_attendue} octets)")
    except Exception:
        # Connexion coupée, refus HTTP, disque plein... : pas de fichier temporaire abandonné
        try:
            os.remove(temporaire)
        except OSError:
            pass
        raise
    os.replace(temporaire, destination)
    return taille


def verifier_mise_a_jour_modele():
    """Compare la version des poids publiée sur le VPS (GET /api/models_ia/latest) à la version
    locale (weights/.current_version). Si elle diffère : télécharge le .pt et le data.yaml sous
    weights/new_best.pt et weights/new_data.yaml, enregistre la nouvelle version, puis arrête
    brutalement le programme. Docker relance alors le conteneur (restart: unless-stopped) et
    docker/entrypoint.sh compile new_best.pt en best.engine (TensorRT, environ 10 min) avant de
    relancer le kiosque. Ne revient pas si une nouvelle version a été installée.
    """
    reponse = requests.get(f"{API_URL}/api/models_ia/latest", headers=EN_TETES_SYNC, timeout=(10, 30))
    if reponse.status_code == 404:
        print("Modèle : aucune version publiée sur le VPS, poids actuels conservés")
        return
    if reponse.status_code != 200:
        print(f"Modèle : refus du serveur (HTTP {reponse.status_code}), poids actuels conservés")
        return

    infos = reponse.json()
    version_serveur = infos["version_name"]
    version_locale = lire_version_locale()
    if version_serveur == version_locale:
        print(f"Modèle : version {version_locale} à jour")
        return

    print("=" * 70)
    print(f"Modèle : NOUVELLE VERSION DISPONIBLE SUR LE VPS : {version_serveur} (en place : {version_locale})")
    print("Modèle : téléchargement des nouveaux poids en cours...")
    try:
        telecharger_fichier(f"{API_URL}{infos['yaml_url']}", NOUVEAU_YAML)
        taille_pt = telecharger_fichier(f"{API_URL}{infos['pt_url']}", NOUVEAU_PT)
    except Exception:
        # Jamais de paire incomplète sur disque (ex. new_data.yaml reçu mais new_best.pt refusé) :
        # l'entrypoint ne compile que si new_best.pt existe, et il doit aller avec son data.yaml
        for chemin in (NOUVEAU_PT, NOUVEAU_YAML):
            try:
                os.remove(chemin)
            except OSError:
                pass
        raise
    print(f"Modèle : {os.path.basename(NOUVEAU_PT)} ({taille_pt / 1e6:.1f} Mo) et {os.path.basename(NOUVEAU_YAML)} reçus")

    ecrire_version_locale(version_serveur)
    print(f"Modèle : version {version_serveur} enregistrée dans {FICHIER_VERSION}")
    print("Modèle : ARRÊT DU KIOSQUE. Au redémarrage du conteneur, l'entrypoint compile le moteur TensorRT (environ 10 min, écran noir pendant ce temps).")
    print("=" * 70)
    sys.stdout.flush()   # os._exit ne vide pas les tampons : sans ceci, les lignes ci-dessus peuvent être perdues
    os._exit(0)


def sync_worker():
    """Envoie périodiquement les captures locales vers le VPS et vérifie, une fois par heure,
    si une nouvelle version des poids du modèle a été publiée.

    Le dossier DOSSIER_ANNOTATION sert de tampon local : une image n'est
    supprimée qu'après confirmation (HTTP 201) de sa réception par le serveur.
    En cas de coupure réseau, tout reste sur disque et sera renvoyé plus tard.
    """
    global reseau_ok
    derniere_verif_modele = 0.0   # 0 = première vérification dès le premier cycle (au démarrage)
    while True:
        try:
            maintenant = time.time()
            images = sorted(
                nom for nom in os.listdir(DOSSIER_ANNOTATION)
                if nom.lower().endswith(".jpg")
                and maintenant - os.path.getmtime(os.path.join(DOSSIER_ANNOTATION, nom)) > SYNC_AGE_MIN
            )
            if SYNC_MAX_IMAGES_PAR_LOT > 0:
                images = images[:SYNC_MAX_IMAGES_PAR_LOT]

            if images and not SYNC_CONFIG_OK:
                reseau_ok = False

            elif images:
                # Archive ZIP en mémoire (JPEG déjà compressé -> ZIP_STORED, pas de recompression inutile)
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_STORED) as archive:
                    for nom in images:
                        archive.write(os.path.join(DOSSIER_ANNOTATION, nom), arcname=nom)
                zip_buffer.seek(0)

                reponse = requests.post(
                    f"{API_URL}/api/sync/upload",
                    files={"archive": ("captures.zip", zip_buffer, "application/zip")},
                    headers=EN_TETES_SYNC,
                    timeout=(10, 180),
                )

                if reponse.status_code == 201:
                    # Le serveur a bien reçu le lot : on libère le tampon local
                    for nom in images:
                        try:
                            os.remove(os.path.join(DOSSIER_ANNOTATION, nom))
                        except OSError as e:
                            print(f"Sync : impossible de supprimer {nom} ({e})")
                    reseau_ok = True
                    print(f"Sync : {len(images)} image(s) envoyée(s) au VPS et supprimée(s) localement")
                else:
                    reseau_ok = False
                    print(f"Sync : refus du serveur (HTTP {reponse.status_code}), tampon local conservé")

        except requests.exceptions.RequestException as e:
            # Timeout, DNS, connexion refusée... : rien n'est supprimé, on réessaiera
            reseau_ok = False
            print(f"Sync : VPS injoignable ({type(e).__name__}), tampon local conservé")
        except Exception as e:
            reseau_ok = False
            print(f"Sync : erreur inattendue ({type(e).__name__}: {e}), tampon local conservé")

        # --- MISE À JOUR DU MODÈLE : une vérification toutes les MODELE_INTERVALLE secondes ---
        # Placée après l'envoi des captures : le tampon local est vidé avant un éventuel redémarrage.
        if SYNC_CONFIG_OK and time.time() - derniere_verif_modele >= MODELE_INTERVALLE:
            derniere_verif_modele = time.time()
            try:
                verifier_mise_a_jour_modele()   # ne revient pas si une nouvelle version a été installée (os._exit)
            except requests.exceptions.HTTPError as e:
                print(f"Modèle : refus du serveur au téléchargement (HTTP {e.response.status_code}), "
                      f"nouvelle tentative dans {MODELE_INTERVALLE // 60} min")
            except requests.exceptions.RequestException as e:
                print(f"Modèle : VPS injoignable ou téléchargement interrompu ({type(e).__name__}), "
                      f"nouvelle tentative dans {MODELE_INTERVALLE // 60} min")
            except Exception as e:
                print(f"Modèle : erreur ({type(e).__name__}: {e}), nouvelle tentative dans {MODELE_INTERVALLE // 60} min")

        time.sleep(SYNC_INTERVALLE)

print("Interface lancée. Appuyez sur vos boutons pour tester (Ctrl+C dans le terminal pour quitter complètement).")

# Lancement de la synchronisation VPS en tâche de fond (thread daemon : s'arrête avec le script)
threading.Thread(target=sync_worker, name="sync_worker", daemon=True).start()

# ==========================================
# 4. BOUCLE PRINCIPALE
# ==========================================
try:
    while running:
        # --- GESTION DES BOUTONS VIA L'ENCODEUR ---
        pygame.event.pump() 

        if joystick is not None:
            for i in range(3):
                etat_actuel = joystick.get_button(i)
                
                # Détection d'un NOUVEL appui (Front montant)
                if etat_actuel and not etat_precedent_boutons[i]:
                    if i == 0 and not en_veille:  # BOUTON 0 : CAPTURE MANUELLE
                        nom_fichier = os.path.join(DOSSIER_ANNOTATION, f"manuel_{datetime.now():%Y%m%d_%H%M%S}.jpg")
                        # On utilisera la frame_brute lue plus bas
                        message_temporaire = "CAPTURE MANUELLE"
                        fin_message = time.time() + 2.0
                        print("Action : Capture manuelle sauvegardée")
                    
                    elif i == 1 and not en_veille:  # BOUTON 1 : MASQUER/AFFICHER IA
                        afficher_ia = not afficher_ia
                        print(f"Action : Affichage IA {'ON' if afficher_ia else 'OFF'}")
                    
                    elif i == 2:  # BOUTON 2 : VEILLE / REVEIL
                        en_veille = not en_veille
                        if en_veille:
                            print("Action : Mode VEILLE activé")
                        else:
                            print("Action : Mode REVEIL activé")
                            
                etat_precedent_boutons[i] = etat_actuel

        # --- MODE VEILLE ---
        if en_veille:
            if fenetre_ouverte:
                cv2.destroyAllWindows()
                fenetre_ouverte = False
            # On lit la caméra dans le vide pour vider le buffer et éviter les latences au réveil
            cap.read()
            time.sleep(0.03)  # Évite d'utiliser 100% du CPU pendant la veille
            continue
        
        # --- MODE REVEIL (Restauration de la fenêtre si besoin) ---
        else:
            if not fenetre_ouverte:
                ouvrir_fenetre()

        # --- LECTURE VIDÉO ---
        ret, frame_brute = cap.read()
        if not ret:
            break

        # Si le bouton 0 a été pressé à cette boucle, on sauvegarde maintenant qu'on a l'image
        if message_temporaire == "CAPTURE MANUELLE" and time.time() < fin_message and (fin_message - time.time()) > 1.9:
            cv2.imwrite(nom_fichier, frame_brute)

        # --- INFÉRENCE IA ---
        results = model(frame_brute, verbose=False)
        alerte_en_cours = len(results[0].boxes) > 0

        # --- SAUVEGARDE AUTOMATIQUE SUR ALERTE ---
        if alerte_en_cours and not alerte_precedente:
            # Cooldown de 3 secondes minimum entre deux captures automatiques
            if time.time() - temps_derniere_alerte > 3.0:
                nom_fichier = os.path.join(DOSSIER_ANNOTATION, f"auto_{datetime.now():%Y%m%d_%H%M%S}.jpg")
                cv2.imwrite(nom_fichier, frame_brute)
                temps_derniere_alerte = time.time()
                message_temporaire = "ALERTE SAUVEGARDEE"
                fin_message = time.time() + 2.0
                print("Action : Alerte détectée -> Capture automatique")
        
        alerte_precedente = alerte_en_cours

        # --- RENDU VISUEL ---
        if afficher_ia:
            frame_affichage = results[0].plot() 
        else:
            frame_affichage = frame_brute.copy()

        couleur_bordure = (0, 0, 255) if alerte_en_cours else (0, 170, 0)
        frame_affichage = cv2.copyMakeBorder(frame_affichage, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=couleur_bordure)

        hauteur, largeur = frame_affichage.shape[:2]
        overlay = frame_affichage.copy()
        cv2.rectangle(overlay, (0, hauteur - 60), (largeur, hauteur), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame_affichage, 0.3, 0, frame_affichage)

        texte_ui = "[B0] Capture Manuelle  |  [B1] Vue IA: " + ("ON" if afficher_ia else "OFF") + "  |  [B2] Veille/Reveil"
        cv2.putText(frame_affichage, texte_ui, (30, hauteur - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if time.time() < fin_message:
            cv2.putText(frame_affichage, message_temporaire, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        if not reseau_ok:
            texte_hors_ligne = "HORS LIGNE - ENREGISTREMENT LOCAL"
            (larg_txt, haut_txt), _ = cv2.getTextSize(texte_hors_ligne, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            cv2.rectangle(frame_affichage, (20, 90), (40 + larg_txt, 110 + haut_txt), (0, 0, 0), -1)
            cv2.putText(frame_affichage, texte_hors_ligne, (30, 100 + haut_txt), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow(nom_fenetre, frame_affichage)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            running = False

except KeyboardInterrupt:
    print("\nFermeture demandée via le terminal.")

cap.release()
cv2.destroyAllWindows()
pygame.quit()
