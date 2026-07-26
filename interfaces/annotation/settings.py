from pathlib import Path
import sys

file_path = Path(__file__).resolve()
app_dir = file_path.parent               # interfaces/annotation/
if str(app_dir) not in sys.path:
    sys.path.append(str(app_dir))
# Chemin absolu : permet d'importer settings/helper depuis n'importe quel
# répertoire courant (ex. serveur mobile), pas seulement via `streamlit run`
# lancé à la racine du projet.
# ROOT = racine du dépôt : les données (dataset_recolte/, exports/) et le
# modèle déployé (weights/) y restent, seul le code vit dans interfaces/.
ROOT = file_path.parents[2]

MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'best.pt'

WEBCAM_PATH = 0

# Le modèle actuel (8 classes) ne détecte que des intrus : aucune classe
# compostable. La liste reste en place pour un futur modèle avec « Organique ».
COMPOSTABLE = []

NON_COMPOSTABLE = ['Plastique', 'Carton', 'Composite', 'Éponge']

MATIERE_RISQUEE = ['Métal', 'Aluminium']

DANGEREUX = ['Verre', 'Céramique']