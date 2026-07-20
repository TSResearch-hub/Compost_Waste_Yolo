from pathlib import Path
import sys

file_path = Path(__file__).resolve()
root_path = file_path.parent
if root_path not in sys.path:
    sys.path.append(str(root_path))
# Chemin absolu : permet d'importer settings/helper depuis n'importe quel
# répertoire courant (ex. serveur mobile), pas seulement via `streamlit run`
# lancé à la racine du projet.
ROOT = root_path

MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'best.pt'

WEBCAM_PATH = 0

# Le modèle actuel (8 classes) ne détecte que des intrus : aucune classe
# compostable. La liste reste en place pour un futur modèle avec « Organique ».
COMPOSTABLE = []

NON_COMPOSTABLE = ['Plastique', 'Carton', 'Composite', 'Éponge']

MATIERE_RISQUEE = ['Métal', 'Aluminium']

DANGEREUX = ['Verre', 'Céramique']