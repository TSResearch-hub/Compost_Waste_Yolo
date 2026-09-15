import os
import time
import subprocess
import cv2

# Masquer les avertissements de polices Qt
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts.warning=false"

# Initialisation de la caméra USB
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la caméra.")
    exit()

# Force le format MJPEG (30 FPS fluides)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

nom_fenetre = "CamPleinEcran"

# 1. Créer une fenêtre brute sans barre de titre ni décorations Qt/Ubuntu
cv2.namedWindow(nom_fenetre, cv2.WINDOW_GUI_NORMAL)
cv2.setWindowProperty(nom_fenetre, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Lire une première image et l'afficher pour initialiser la fenêtre au niveau système
ret, frame = cap.read()
if ret:
    cv2.imshow(nom_fenetre, frame)
    cv2.waitKey(100)

# 2. Forcer Ubuntu (via wmctrl) à masquer la barre Ubuntu et la barre de titre
try:
    # Retire les décorations de fenêtres et applique l'état "Fullscreen" d'Ubuntu
    subprocess.run(["wmctrl", "-r", nom_fenetre, "-b", "add,fullscreen,above"], check=False)
except Exception as e:
    print(f"Attention : impossible d'exécuter wmctrl : {e}")

print("Affichage en plein écran total. Appuyez sur 'q' pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur de lecture du flux vidéo.")
        break

    # Affichage de la vidéo
    cv2.imshow(nom_fenetre, frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
