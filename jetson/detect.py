import os
import subprocess
import cv2
from ultralytics import RTDETR

# Masquer les avertissements de polices Qt
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts.warning=false"

# 1. CHARGEMENT DU MODÈLE OPTIMISÉ TENSORRT
print("Chargement du moteur TensorRT... L'initialisation est très rapide.")
model = RTDETR("weights/best.engine")

# Initialisation de la caméra USB
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la caméra.")
    exit()

# Force le format MJPEG (30 FPS fluides) et la résolution
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

nom_fenetre = "CamPleinEcran"

# Créer une fenêtre brute sans barre de titre
cv2.namedWindow(nom_fenetre, cv2.WINDOW_GUI_NORMAL)
cv2.setWindowProperty(nom_fenetre, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Lire une première image pour initialiser la fenêtre
ret, frame = cap.read()
if ret:
    cv2.imshow(nom_fenetre, frame)
    cv2.waitKey(100)

# Forcer Ubuntu (via wmctrl) à masquer la barre Ubuntu et la barre de titre
try:
    subprocess.run(["wmctrl", "-r", nom_fenetre, "-b", "add,fullscreen,above"], check=False)
except Exception as e:
    print(f"Attention : impossible d'exécuter wmctrl : {e}")

print("Affichage avec détection TensorRT en cours. Appuyez sur 'q' pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur de lecture du flux vidéo.")
        break

    # 2. INFÉRENCE AVEC TENSORRT
    # L'inférence sera beaucoup plus rapide qu'avant
    results = model(frame, verbose=False)

    # 3. DESSIN DES RÉSULTATS
    annotated_frame = results[0].plot()

    # Affichage de la vidéo annotée
    cv2.imshow(nom_fenetre, annotated_frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
