from pathlib import Path
import sys

file_path = Path(__file__).resolve()
root_path = file_path.parent
if root_path not in sys.path:
    sys.path.append(str(root_path))
ROOT = root_path.relative_to(Path.cwd())

MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'best.pt'

WEBCAM_PATH = 2

COMPOSTABLE = ['Compost'] 

NON_COMPOSTABLE = ['NonCompost']

MATIERE_RISQUEE = ['Mrisq']

DANGEREUX = ['Dgrx']

RECYCLABLE = COMPOSTABLE
NON_RECYCLABLE = NON_COMPOSTABLE