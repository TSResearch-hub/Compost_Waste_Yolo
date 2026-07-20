from pathlib import Path
import os
import tempfile
import streamlit as st
import helper
import settings
import annotation_timer
import dataset_tools
import cv2
from PIL import Image
from bbox_editor import detection as st_detection
import time
import numpy as np

st.set_page_config(
    page_title="Composte IA",
    layout="wide",
    initial_sidebar_state="collapsed",   # Sidebar fermée par défaut → max espace
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS — TABLEAU DE BORD INDUSTRIEL
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ══ CHROME STREAMLIT — tout masqué, mode WebApp industrielle ════════════ */
#MainMenu { visibility: hidden; }
header    { visibility: hidden; }
footer    { visibility: hidden; }

/* ══ DENSITÉ MAXIMALE — l'image doit occuper tout l'espace ══════════════ */
.block-container {
    padding-top: 0.6rem !important;
    padding-bottom: 0.6rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    max-width: 95% !important;
}

/* ══ TOUS LES BOUTONS — base tactile / gants ══════════════════════════════ */
div[data-testid="stButton"] > button {
    min-height: 2.8rem !important;
    font-size: 0.97rem !important;
    border-radius: 0.5rem !important;
    font-weight: 600 !important;
    transition: opacity 0.15s;
}

/* ══ BOUTON PRIMARY — dominant, immanquable ══════════════════════════════ */
div[data-testid="stButton"] > button[kind="primary"] {
    min-height: 4rem !important;
    font-size: 1.25rem !important;
    letter-spacing: 0.04em;
}

/* ══ BOUTONS DESTRUCTIFS — Effacer tout + Vider la file ══════════════════ */
button[aria-label="🗑 Effacer tout"],
button[aria-label="🗑 Vider"] {
    min-height: 2.2rem !important;
    font-size: 0.82rem !important;
    opacity: 0.75;
    border: 1.5px solid #c23232 !important;
    color: #c23232 !important;
}
button[aria-label="🗑 Effacer tout"]:hover,
button[aria-label="🗑 Vider"]:hover {
    background-color: #c23232 !important;
    color: #fff !important;
    opacity: 1 !important;
}

/* ══ RADIO BUTTONS — lisibles, espacement tactile ════════════════════════ */
div[data-testid="stRadio"] label {
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 0.3rem 0.2rem !important;
}
div[data-testid="stRadio"] > div { gap: 0.6rem !important; }

/* ══ CONTENEUR ÉDITEUR — bordure subtile pour délimiter la zone de travail */
.editor-container {
    border: 1px solid rgba(150,150,170,0.18);
    border-radius: 0.75rem;
    padding: 0.9rem 1.2rem 1.2rem;
    margin-top: 0.5rem;
    background: rgba(10,10,20,0.025);
}
/* Centrage CSS du canvas — remplace la colonne [1,4,1] supprimée
   (qui causait 3 niveaux d'imbrication interdits par Streamlit).  */
.editor-canvas-wrap {
    display: flex;
    justify-content: center;
    margin: 0.3rem 0;
}

/* ══ ALERTES "FEU TRICOLORE INDUSTRIEL" ══════════════════════════════════ */
.alert-block {
    display: flex;
    align-items: center;
    gap: 0.85rem;
    padding: 1rem 1.1rem;
    border-radius: 0.65rem;
    margin-bottom: 0.75rem;
    color: #fff;
    box-shadow: 0 3px 10px rgba(0,0,0,0.22);
}
.alert-icon { font-size: 2rem; line-height: 1; flex-shrink: 0; }
.alert-block strong {
    display: block; font-size: 1rem; font-weight: 800;
    letter-spacing: 0.04em; text-transform: uppercase;
}
.alert-block small { display: block; font-size: 0.82rem; margin-top: 0.15rem; opacity: 0.9; }
.alert-danger      { background: linear-gradient(135deg, #8b0000, #c23232); border-left: 6px solid #ff5555; }
.alert-risk        { background: linear-gradient(135deg, #b85e00, #ff8c00); border-left: 6px solid #ffb347; }
.alert-non-compost { background: linear-gradient(135deg, #2e4d74, #5e80ad); border-left: 6px solid #89b4e8; }
.alert-compost     { background: linear-gradient(135deg, #0a6620, #16b939); border-left: 6px solid #5dde7f; }
.alert-idle {
    padding: 1.4rem 1rem; border-radius: 0.65rem;
    background: rgba(100,100,120,0.10);
    border: 2px dashed rgba(150,150,170,0.3);
    color: rgba(140,140,160,0.9);
    text-align: center; font-size: 0.88rem; line-height: 1.6;
}
section[data-testid="stSidebar"] .stMarkdown p { font-size: 0.85rem; }

/* ══ ONGLET DATASET — barres de distribution par classe ══════════════════ */
/* Couleurs identiques aux bboxes de l'éditeur (identité par entité) ; le
   nom + l'effectif sont toujours écrits en toutes lettres à côté de la barre,
   la couleur seule ne porte jamais l'information. */
.dist-row {
    display: grid;
    grid-template-columns: 130px 1fr 110px;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 2px;
}
.dist-name {
    font-size: 0.85rem; font-weight: 600;
    display: flex; align-items: center; gap: 0.45rem;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.dist-swatch { width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0; }
.dist-track  { background: rgba(128,128,144,0.13); border-radius: 0 4px 4px 0; height: 14px; }
.dist-bar    { height: 100%; border-radius: 0 4px 4px 0; }
.dist-count  {
    font-size: 0.82rem; opacity: 0.85; text-align: right;
    font-variant-numeric: tabular-nums; white-space: nowrap;
}
.dist-zero .dist-name, .dist-zero .dist-count { opacity: 0.45; }

/* ══ RESPONSIVE — TABLETTES & SMARTPHONES ══════════════════════════════════ */

/* Tablette portrait (≤ 900 px) */
@media (max-width: 900px) {
    .block-container {
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        max-width: 100% !important;
    }
    /* Colonnes Streamlit : autoriser le retour à la ligne */
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        gap: 0.5rem 0 !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        min-width: min(100%, 280px) !important;
        flex: 1 1 auto !important;
    }
    /* Tabs : texte un peu plus compact */
    [data-testid="stTabBar"] button p {
        font-size: 0.82rem !important;
    }
}

/* Smartphone portrait (≤ 600 px) */
@media (max-width: 600px) {
    /* Toutes les colonnes passent en pleine largeur */
    [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }
    /* Tabs : icône + texte court pour tenir sur une ligne */
    [data-testid="stTabBar"] {
        overflow-x: auto !important;
        flex-wrap: nowrap !important;
    }
    [data-testid="stTabBar"] button p {
        font-size: 0.72rem !important;
        white-space: nowrap !important;
    }
    /* Boutons tactiles plus généreux */
    div[data-testid="stButton"] > button {
        min-height: 3.2rem !important;
        font-size: 1rem !important;
    }
    div[data-testid="stButton"] > button[kind="primary"] {
        min-height: 3.5rem !important;
        font-size: 1.1rem !important;
    }
    /* Alertes : un peu plus compactes */
    .alert-block {
        padding: 0.65rem 0.75rem !important;
        gap: 0.55rem !important;
    }
    .alert-icon { font-size: 1.5rem !important; }
    .alert-block strong { font-size: 0.88rem !important; }
    .alert-block small  { font-size: 0.74rem !important; }
    /* Metrics vidéo (durée, fps, frames) */
    [data-testid="stMetric"] {
        padding: 0.4rem !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.2rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DÉTECTION DES CAMÉRAS DISPONIBLES
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Recherche des caméras...")
def scan_available_cameras(max_index: int = 6) -> list:
    """
    Détecte les indices de caméra disponibles (0 à max_index-1).
    Le résultat est mis en cache pour ne pas rescanner à chaque rerun.
    """
    available = []
    for i in range(max_index):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
            cap.release()
        except Exception:
            pass
    return available if available else [0]


# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DU MODÈLE
# ══════════════════════════════════════════════════════════════════════════════
model_path = Path(settings.DETECTION_MODEL)
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Impossible de charger le modèle : {model_path}")
    st.error(ex)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════
SAVE_DIR = settings.ROOT / "dataset_recolte"
LABEL_LIST = list(helper.CLASS_MAP.keys())
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Côté max de l'image DANS L'ÉDITEUR : les photos 3000×4000 rendaient Konva
# poussif. Le ratio est conservé et les labels YOLO sont normalisés (0-1) avec
# les dimensions de l'image affichée, donc les coordonnées restent exactes pour
# l'image originale — qui, elle, est toujours sauvegardée pleine résolution.
MAX_EDIT_SIDE = 1500

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — réglages profonds + stats du dataset
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ Paramètres")
    conf_threshold = st.slider("Seuil de confiance", 0.1, 0.95, 0.4, 0.05)

    # Seuils de pré-annotation par classe : utile quand le modèle est fiable
    # sur une classe (seuil haut = moins de faux positifs à supprimer) et
    # hésitant sur une autre (seuil bas = moins de bboxes à dessiner à la main).
    with st.expander("🎚 Seuils par classe (pré-annotation)"):
        per_class_on = st.checkbox("Personnaliser par classe", key="per_class_conf_on")
        st.caption(
            "Appliqués aux pré-annotations de l'éditeur (hors ligne, vérification, "
            "captures). Le direct garde le seuil global. Modifiable en cours de lot : "
            "utilisez **🤖 Re-pré-annoter** sous l'éditeur pour ré-appliquer."
        )
        class_conf = {
            cid: st.slider(name, 0.05, 0.95, conf_threshold, 0.05,
                           key=f"conf_cls_{cid}", disabled=not per_class_on)
            for name, cid in helper.CLASS_MAP.items()
        }

    # Sélection caméra par liste (plus d'index manuel aveugle)
    cameras = scan_available_cameras()
    camera_index = st.selectbox(
        "Caméra disponible",
        options=cameras,
        format_func=lambda i: f"Caméra {i}",
    )
    if st.button("🔄 Rescanner les caméras", use_container_width=True):
        scan_available_cameras.clear()
        st.rerun()

    st.divider()
    st.subheader("📊 Dataset")
    _n_labels = len(list((SAVE_DIR / "labels").glob("*.txt"))) if (SAVE_DIR / "labels").exists() else 0
    _n_images = (
        len([p for p in (SAVE_DIR / "images").iterdir() if p.suffix.lower() in IMG_EXTENSIONS])
        if (SAVE_DIR / "images").exists() else 0
    )
    _c1, _c2 = st.columns(2)
    _c1.metric("Images", _n_images)
    _c2.metric("Annotées", _n_labels)
    _stats = annotation_timer.get_stats()
    if _stats:
        _line = f"Aujourd'hui : **{_stats['today']}** annotation(s)"
        if _stats["mean_sec_today"] is not None:
            _line += f" · ⏱ moy. **{_stats['mean_sec_today']:.0f} s**/image"
        st.caption(_line)
        st.caption(f"Total historique : {_stats['total']} annotations")

    st.divider()
    st.caption("Composte IA · Station de compostage")

# Seuils effectifs de pré-annotation : par classe si activé, sinon le global.
CONF_BY_ID = class_conf if per_class_on else {cid: conf_threshold for cid in helper.CLASS_MAP.values()}
# Le predict tourne sous le plus bas des seuils : on récupère large une fois,
# le tri fin se fait ensuite par classe (et reste re-jouable sans re-predict
# si les seuils changent en cours de route).
PRED_CONF = min(0.1, *CONF_BY_ID.values())

# ══════════════════════════════════════════════════════════════════════════════
# UTILITAIRES CANVAS & SAUVEGARDE
# ══════════════════════════════════════════════════════════════════════════════

def _prepare_save_dirs():
    img_dir = SAVE_DIR / "images"
    lbl_dir = SAVE_DIR / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    return img_dir, lbl_dir


def _reset_canvas_state(canvas_key):
    """Supprime tout l'état lié à ce canvas_key."""
    for suffix in ("_data", "_clear_counter", "_img_id"):
        st.session_state.pop(f"{canvas_key}{suffix}", None)


def _resolve_name_collision(img_dir, stem, ext, raw_bytes):
    """
    Évite d'écraser silencieusement une image déjà présente dans le dataset.
    - Fichier absent                          → stem inchangé.
    - Fichier présent avec le même contenu    → stem inchangé (ré-annotation
      volontaire de la même image : le label sera simplement mis à jour).
    - Fichier présent avec un contenu différent (collision de nom entre deux
      lots d'import) → suffixe incrémental stem_2, stem_3, ...
    """
    candidate = stem
    i = 1
    while (img_dir / f"{candidate}{ext}").exists():
        if raw_bytes is not None and (img_dir / f"{candidate}{ext}").read_bytes() == raw_bytes:
            return candidate
        i += 1
        candidate = f"{stem}_{i}"
    return candidate


def _save_annotation(pil_img, detection_result, w_img, h_img, original_name=None, raw_bytes=None, overwrite=False):
    """
    Sauvegarde l'image et le fichier YOLO depuis le résultat du composant detection().
    detection_result : liste de {'bbox': [x, y, w, h], 'label_id': int, 'label': str}
    Les coordonnées bbox sont en pixels dans l'espace image original.
    Si raw_bytes est fourni, les bytes originaux sont écrits tels quels (sans réencodage).
    overwrite=True (onglet Vérification) : on réécrit volontairement l'annotation existante.
    """
    img_dir, lbl_dir = _prepare_save_dirs()

    if original_name:
        stem = Path(original_name).stem
        ext  = Path(original_name).suffix.lower() or ".jpg"
    else:
        stem = f"cap_{int(time.time())}"
        ext  = ".jpg"

    if not overwrite:
        stem = _resolve_name_collision(img_dir, stem, ext, raw_bytes)

    if raw_bytes is not None:
        with open(img_dir / f"{stem}{ext}", "wb") as fout:
            fout.write(raw_bytes)
    else:
        pil_img.save(img_dir / f"{stem}{ext}")

    yolo_lines = []
    for item in detection_result:
        x, y, w_box, h_box = item["bbox"]
        x_center = (x + w_box / 2) / w_img
        y_center = (y + h_box / 2) / h_img
        norm_w = w_box / w_img
        norm_h = h_box / h_img
        yolo_lines.append(
            f"{item['label_id']} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
        )

    with open(lbl_dir / f"{stem}.txt", "w") as f:
        f.write("\n".join(yolo_lines))

    return f"{stem}{ext}"


def _parse_yolo_label(text, w_img, h_img):
    """Parse un fichier YOLO (classe xc yc w h normalisés) → bboxes pixels + indices."""
    bboxes, labels_idx = [], []
    for line in text.strip().split("\n"):
        parts = line.strip().split()
        if len(parts) == 5:
            try:
                cid = int(parts[0])
                xc, yc, bw, bh = (float(p) for p in parts[1:])
                bboxes.append(
                    [(xc - bw / 2) * w_img, (yc - bh / 2) * h_img, bw * w_img, bh * h_img]
                )
                labels_idx.append(min(cid, len(LABEL_LIST) - 1))
            except ValueError:
                pass
    return bboxes, labels_idx


def _exit_annotation(canvas_key, exit_mode_annotation, offline_mode=False):
    """
    Post-sauvegarde/annulation : enchaîne automatiquement sur l'image suivante.
    - offline_mode=True  → avance vers la prochaine image non annotée d'offline_items
    - offline_mode=False → avance dans capture_queue (webcam)
    """
    st.session_state["_annot_token"] = st.session_state.get("_annot_token", 0) + 1
    _reset_canvas_state(canvas_key)
    if not exit_mode_annotation:
        return  # Cas hors ligne image unique (legacy)

    if offline_mode:
        # Avance vers la prochaine image non annotée du lot (après l'image
        # courante, sinon la première restante) ; le lot reste navigable.
        items = st.session_state.get("offline_items", [])
        idx = st.session_state.get("offline_idx", 0)
        pending = [i for i, it in enumerate(items) if not it.get("saved_name")]
        if pending:
            st.session_state["offline_idx"] = next((i for i in pending if i > idx), pending[0])
    else:
        # Mode webcam : avance dans capture_queue
        queue = st.session_state.get("capture_queue", [])
        if queue:
            next_item = st.session_state["capture_queue"].pop(0)
            st.session_state["last_raw_image"] = next_item["image"]
            st.session_state["last_res"] = next_item["res"]
            st.session_state["mode_annotation"] = True
        else:
            st.session_state["mode_annotation"] = False
    st.rerun()


def _offline_goto(i):
    """Navigation manuelle dans le lot hors ligne (⬅/➡/sélecteur)."""
    st.session_state["offline_idx"] = i
    st.session_state["_annot_token"] = st.session_state.get("_annot_token", 0) + 1
    _reset_canvas_state("canvas_offline")
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ÉDITEUR D'ANNOTATION
# ══════════════════════════════════════════════════════════════════════════════

def show_annotation_editor(raw_img, last_res, canvas_key, exit_mode_annotation=True, offline_mode=False,
                           original_name=None, raw_bytes=None, initial_data=None, conf_by_id=None,
                           overwrite=False, on_saved=None):
    """
    Éditeur d'annotation YOLO basé sur streamlit_image_annotation (Konva).

    Le composant gère nativement :
      - Mode Transform : dessiner de nouvelles bbox + sélectionner / déplacer / redimensionner
      - Mode Del       : clic sur une bbox pour la supprimer
      - Sélecteur de classe : dropdown intégré
      - Bouton "Complete" : envoie l'état final → déclenche la sauvegarde côté Python

    Boutons Streamlit complémentaires : 🗑 Effacer tout | ✖ Annuler
    """
    h_img, w_img = raw_img.shape[:2]
    img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb).convert("RGB")

    _imgid_key   = f"{canvas_key}_img_id"
    _data_key    = f"{canvas_key}_data"
    _counter_key = f"{canvas_key}_clear_counter"

    # Réinitialise les bboxes quand l'image change.
    # initial_data (bboxes, labels) prime sur la pré-annotation du modèle :
    # sert à recharger une annotation déjà sauvegardée (revisite dans un lot).
    if st.session_state.get(_imgid_key) != id(raw_img):
        st.session_state[_imgid_key]   = id(raw_img)
        if initial_data is not None:
            bboxes, labels = initial_data
        else:
            bboxes, labels = helper.get_detection_initial_data(last_res, conf_by_id)
        st.session_state[_data_key]    = {"bboxes": bboxes, "labels": labels}
        st.session_state[_counter_key] = 0
        annotation_timer.start_timer(canvas_key)

    clear_counter = st.session_state.get(_counter_key, 0)
    annot_token   = st.session_state.get("_annot_token", 0)
    data = st.session_state[_data_key]
    source = "offline" if offline_mode else "webcam"

    st.markdown("<div class='editor-container'>", unsafe_allow_html=True)

    # ── Canvas Konva via bbox_editor ──────────────────────────────────────────
    # token est incrémenté à chaque changement d'image dans _exit_annotation().
    # Le composant React renvoie {token, bboxes:[...]} ; si le token ne correspond
    # pas, Python ignore la valeur (valeur mise en cache Streamlit d'une ancienne image).
    result = st_detection(
        image=pil_img,
        label_list=LABEL_LIST,
        bboxes=data["bboxes"],
        labels=data["labels"],
        height=h_img,
        width=w_img,
        line_width=2,
        use_space=False,
        color_map=helper.CLASS_COLORS,
        token=annot_token,
        image_name=original_name or "",
        start_time_ms=annotation_timer.get_start_time_ms(canvas_key),
        key=f"{canvas_key}_{annot_token}_{clear_counter}",
    )

    # ── Sauvegarde déclenchée par "Complete" dans le composant ────────────────
    if result is not None:
        saved_name = _save_annotation(pil_img, result, w_img, h_img, original_name=original_name,
                                      raw_bytes=raw_bytes, overwrite=overwrite)
        if on_saved is not None:
            on_saved(saved_name)
        log_row = annotation_timer.log_annotation(canvas_key, image_name=saved_name, source=source, nb_boxes=len(result))
        st.toast(f"Annotation sauvegardée ! (⏱ {log_row['duration_display']})", icon="✅")
        _exit_annotation(canvas_key, exit_mode_annotation, offline_mode=offline_mode)

    # ── Boutons complémentaires ───────────────────────────────────────────────
    st.write("")
    col_clear, col_reannot, col_cancel = st.columns([1, 1, 1])

    with col_clear:
        if st.button("🗑 Effacer tout", key=f"clear_{canvas_key}", use_container_width=True):
            st.session_state[_data_key]    = {"bboxes": [], "labels": []}
            st.session_state[_counter_key] = clear_counter + 1
            st.rerun()

    with col_reannot:
        # Ré-applique la pré-annotation du modèle avec les seuils ACTUELS de la
        # sidebar : permet d'ajuster les seuils par classe en cours de lot.
        if last_res is not None and st.button("🤖 Re-pré-annoter", key=f"reannot_{canvas_key}", use_container_width=True):
            bboxes, labels = helper.get_detection_initial_data(last_res, conf_by_id)
            st.session_state[_data_key]    = {"bboxes": bboxes, "labels": labels}
            st.session_state[_counter_key] = clear_counter + 1
            st.rerun()

    with col_cancel:
        if st.button("✖ Annuler", key=f"cancel_{canvas_key}", use_container_width=True):
            if exit_mode_annotation:
                annotation_timer.log_annotation(canvas_key, image_name=original_name or "", source=source, nb_boxes=0, status="cancelled")
                _exit_annotation(canvas_key, exit_mode_annotation, offline_mode=offline_mode)

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ONGLETS PRINCIPAUX
# ══════════════════════════════════════════════════════════════════════════════
tab_detection, tab_offline, tab_video, tab_verify, tab_dataset = st.tabs(
    ["📹 Détection en direct", "🗂 Annotation hors ligne", "🎬 Extraction vidéo", "🔍 Vérification", "📊 Dataset"]
)

# ─── ONGLET 1 : DÉTECTION EN DIRECT ──────────────────────────────────────────
with tab_detection:

    # Panneau d'alertes créé EN PREMIER (avant la boucle webcam)
    # pour que le placeholder existe et puisse être mis à jour dans play_webcam.
    col_video, col_alerts = st.columns([7, 3])

    with col_alerts:
        st.subheader("🔍 Détections")
        alert_placeholder = st.empty()
        if not st.session_state.get("run_detection", False):
            alert_placeholder.markdown(
                "<div class='alert-idle'>📡 Caméra inactive<br>"
                "<small>Lancez la détection pour voir les alertes</small></div>",
                unsafe_allow_html=True,
            )

    with col_video:
        with st.expander("⚙️ Options de capture automatique"):
            auto_capture_on = st.checkbox("Activer la capture automatique")
            auto_interval = st.slider("Intervalle (s)", 1, 60, 5) if auto_capture_on else 0

        helper.play_webcam(
            model,
            alert_placeholder=alert_placeholder,
            auto_capture_interval=auto_interval,
            conf=conf_threshold,
            camera_index=camera_index,
        )

    # ── File d'attente ────────────────────────────────────────────────────────
    # Le bouton "Traiter" démarre le premier cycle d'annotation.
    # Ensuite, _exit_annotation() enchaîne automatiquement sans intervention.
    queue = st.session_state.get("capture_queue", [])
    if queue:
        with st.expander(
            f"📥 File d'annotation — {len(queue)} image(s) en attente", expanded=True
        ):
            col_next, col_clear = st.columns([3, 1])
            with col_next:
                if st.button(
                    f"▶ Traiter la file ({len(queue)} image(s))",
                    type="primary",
                    use_container_width=True,
                ):
                    next_item = st.session_state["capture_queue"].pop(0)
                    st.session_state["last_raw_image"] = next_item["image"]
                    st.session_state["last_res"] = next_item["res"]
                    st.session_state["mode_annotation"] = True
                    st.rerun()
            with col_clear:
                if st.button("🗑 Vider", use_container_width=True):
                    st.session_state["capture_queue"] = []
                    st.rerun()

    # ── Éditeur d'annotation (webcam / file d'attente) ────────────────────────
    if st.session_state.get("mode_annotation", False) and "last_raw_image" in st.session_state:
        # Indicateur de progression si des images restent dans la file
        remaining = len(st.session_state.get("capture_queue", []))
        if remaining:
            st.info(f"Image en cours — {remaining} image(s) suivront automatiquement.")
        show_annotation_editor(
            st.session_state["last_raw_image"],
            st.session_state.get("last_res"),
            canvas_key="canvas_webcam",
            exit_mode_annotation=True,
            conf_by_id=CONF_BY_ID,
        )

# ─── ONGLET 2 : ANNOTATION HORS LIGNE ────────────────────────────────────────
with tab_offline:
    offline_items = st.session_state.get("offline_items", [])

    if not offline_items:
        # ── Formulaire d'import ───────────────────────────────────────────────
        st.write("Importez une ou plusieurs images pour les annoter en série.")
        uploaded_files = st.file_uploader(
            "Choisir des images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            if st.button(
                f"▶ Annoter {len(uploaded_files)} image(s)",
                type="primary",
                use_container_width=True,
            ):
                # Décodage + réduction à MAX_EDIT_SIDE pour un éditeur fluide
                # (les bytes originaux pleine résolution restent ce qui est
                # sauvegardé dans le dataset). Prédiction en lazy à l'affichage.
                items, ignored = [], []
                for f in uploaded_files:
                    raw_bytes = f.getvalue()
                    nparr = np.frombuffer(raw_bytes, np.uint8)
                    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img_bgr is None:
                        ignored.append(f.name)
                        continue
                    img_bgr = helper.resize_keep_ratio(img_bgr, MAX_EDIT_SIDE, MAX_EDIT_SIDE)
                    items.append({"image": img_bgr, "name": f.name, "res": None,
                                  "raw_bytes": raw_bytes, "saved_name": None})
                if ignored:
                    st.warning("Image(s) illisible(s) ignorée(s) : " + ", ".join(ignored))
                if items:
                    st.session_state["offline_items"] = items
                    st.session_state["offline_idx"] = 0
                    _reset_canvas_state("canvas_offline")
                    st.rerun()
    else:
        # ── Annotation en série, avec navigation libre dans le lot ────────────
        total  = len(offline_items)
        idx    = min(st.session_state.get("offline_idx", 0), total - 1)
        n_done = sum(1 for it in offline_items if it.get("saved_name"))
        current = offline_items[idx]

        st.progress(n_done / total, text=f"{n_done}/{total} annotée(s) — affichée : {current['name']}")

        col_sel, col_prev, col_next, col_stop = st.columns([4, 1, 1, 2])
        with col_sel:
            pick = st.selectbox(
                "Image du lot",
                range(total),
                index=idx,
                format_func=lambda i: (
                    f"{'✅' if offline_items[i].get('saved_name') else '⬜'} "
                    f"{i + 1}/{total} — {offline_items[i]['name']}"
                ),
                label_visibility="collapsed",
            )
            if pick != idx:
                _offline_goto(pick)
        if col_prev.button("⬅", use_container_width=True, disabled=idx == 0,
                           help="Image précédente"):
            _offline_goto(idx - 1)
        if col_next.button("➡", use_container_width=True, disabled=idx == total - 1,
                           help="Image suivante"):
            _offline_goto(idx + 1)
        if col_stop.button("⏹ Terminer le lot", use_container_width=True):
            st.session_state.pop("offline_items", None)
            st.session_state.pop("offline_idx", None)
            _reset_canvas_state("canvas_offline")
            st.rerun()

        if n_done == total:
            st.success("🏁 Tout le lot est annoté ! Vous pouvez encore revenir corriger "
                       "une image via le sélecteur, ou cliquer **⏹ Terminer le lot**.")

        # Image déjà sauvegardée : on recharge son annotation du dataset pour la
        # corriger (la sauvegarde écrasera proprement la même entrée, sans doublon).
        already = current.get("saved_name")
        initial_data = None
        if already:
            lbl_path = SAVE_DIR / "labels" / f"{Path(already).stem}.txt"
            if lbl_path.exists():
                h_cur, w_cur = current["image"].shape[:2]
                initial_data = _parse_yolo_label(lbl_path.read_text(encoding="utf-8"), w_cur, h_cur)
            st.caption(f"✏️ Déjà annotée (`{already}`) — sauvegarder mettra à jour l'annotation existante.")
        elif current["res"] is None:
            # Prédiction lazy, une seule fois par image, sous le seuil plancher :
            # le filtrage fin par classe se fait à l'affichage (re-jouable).
            with st.spinner("Analyse par le modèle..."):
                current["res"] = model.predict(current["image"], conf=PRED_CONF)[0]
                # La mutation de current met à jour session_state directement
                # car c'est le même objet en mémoire.

        show_annotation_editor(
            current["image"],
            current["res"],
            canvas_key="canvas_offline",
            exit_mode_annotation=True,
            offline_mode=True,
            original_name=already or current.get("name"),
            raw_bytes=current.get("raw_bytes"),
            initial_data=initial_data,
            conf_by_id=CONF_BY_ID,
            overwrite=bool(already),
            on_saved=lambda name, item=current: item.__setitem__("saved_name", name),
        )

# ─── ONGLET 3 : EXTRACTION VIDÉO ─────────────────────────────────────────────
with tab_video:
    st.write("Importez une vidéo pour en extraire des images à intervalle régulier.")
    st.write("Les images extraites seront envoyées directement vers l'onglet **Annotation hors ligne**.")

    uploaded_video = st.file_uploader(
        "Choisir une vidéo",
        type=["mp4", "avi", "mov", "mkv", "webm"],
    )

    if uploaded_video is not None:
        suffix = Path(uploaded_video.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded_video.getvalue())
            tmp_path = tmp.name

        try:
            cap = cv2.VideoCapture(tmp_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            duration = total_frames / fps if fps > 0 else 0

            c1, c2, c3 = st.columns(3)
            c1.metric("Durée", f"{duration:.1f} s")
            c2.metric("FPS", f"{fps:.0f}")
            c3.metric("Frames totales", total_frames)

            max_interval = max(1, int(duration))
            interval = st.slider(
                "Intervalle d'extraction (s)",
                min_value=1,
                max_value=min(max_interval, 120),
                value=min(5, max_interval),
                step=1,
            )
            n_to_extract = max(1, int(duration / interval))
            st.caption(f"→ {n_to_extract} image(s) à extraire")

            if st.button(
                f"🎬 Extraire {n_to_extract} image(s)",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner("Extraction en cours..."):
                    frames = helper.extract_frames_from_video(tmp_path, interval)
                for fr in frames:
                    fr["saved_name"] = None
                st.session_state["offline_items"] = frames
                st.session_state["offline_idx"] = 0
                _reset_canvas_state("canvas_offline")
                st.toast(f"{len(frames)} images extraites — allez dans 'Annotation hors ligne'", icon="🎬")
                st.rerun()
        finally:
            os.unlink(tmp_path)

# ─── ONGLET 4 : VÉRIFICATION D'ANNOTATION ────────────────────────────────────
with tab_verify:
    v_source = st.radio(
        "Source",
        ["📁 Dataset enregistré", "⬆ Import manuel"],
        horizontal=True,
        key="verify_source",
        label_visibility="collapsed",
    )

    # ── Mode 1 : parcours direct de dataset_recolte/ (plus d'upload manuel) ──
    if v_source == "📁 Dataset enregistré":
        img_dir = SAVE_DIR / "images"
        lbl_dir = SAVE_DIR / "labels"
        all_imgs = (
            sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS)
            if img_dir.exists() else []
        )

        if not all_imgs:
            st.info("Aucune image dans `dataset_recolte/images` pour le moment. "
                    "Annotez d'abord quelques images ou utilisez l'import manuel.")
        else:
            has_label = {p.name: (lbl_dir / f"{p.stem}.txt").exists() for p in all_imgs}

            col_f1, col_f2 = st.columns([2, 3])
            with col_f1:
                v_filter = st.selectbox(
                    "Filtre",
                    ["Toutes", "✅ Annotées", "⬜ Sans annotation"]
                    + [f"🏷 {n}" for n in LABEL_LIST],
                    key="verify_filter",
                )
            if v_filter.startswith("🏷 "):
                # Filtre « contient la classe » : pratique pour repasser sur une
                # classe suspecte (ex. les Aluminium issus de l'ancien bug de mapping).
                wanted_cid = LABEL_LIST.index(v_filter.removeprefix("🏷 "))
                cls_by_stem = dataset_tools.label_classes(lbl_dir)
                names = [
                    p.name for p in all_imgs
                    if wanted_cid in cls_by_stem.get(p.stem, set())
                ]
            else:
                names = [
                    p.name for p in all_imgs
                    if v_filter == "Toutes"
                    or (v_filter == "✅ Annotées" and has_label[p.name])
                    or (v_filter == "⬜ Sans annotation" and not has_label[p.name])
                ]

            if not names:
                st.info("Aucune image ne correspond à ce filtre.")
            else:
                # Saut de navigation demandé au run précédent (boutons ⬅/➡ ou
                # auto-avance après sauvegarde) : appliqué AVANT de créer le widget.
                if "_verify_jump" in st.session_state:
                    jump = st.session_state.pop("_verify_jump")
                    if jump in names:
                        st.session_state["verify_select"] = jump
                if st.session_state.get("verify_select") not in names:
                    st.session_state["verify_select"] = names[0]

                with col_f2:
                    v_name = st.selectbox(
                        "Image",
                        names,
                        key="verify_select",
                        format_func=lambda n: f"{'✅' if has_label[n] else '⬜'} {n}",
                    )
                idx = names.index(v_name)

                col_prev, col_pos, col_next, col_auto, col_del = st.columns([1, 1, 1, 2, 1])
                if col_prev.button("⬅ Précédente", use_container_width=True, disabled=idx == 0):
                    st.session_state["_verify_jump"] = names[idx - 1]
                    st.rerun()
                col_pos.markdown(
                    f"<div style='text-align:center;padding-top:0.5rem;'><b>{idx + 1} / {len(names)}</b></div>",
                    unsafe_allow_html=True,
                )
                if col_next.button("Suivante ➡", use_container_width=True, disabled=idx == len(names) - 1):
                    st.session_state["_verify_jump"] = names[idx + 1]
                    st.rerun()
                auto_advance = col_auto.checkbox(
                    "Image suivante après sauvegarde", value=True, key="verify_auto_next"
                )
                # Suppression réversible : image + label partent dans
                # dataset_recolte/corbeille/ (jamais d'effacement définitif).
                with col_del.popover("🗑 Supprimer", use_container_width=True):
                    st.markdown(f"Retirer **{v_name}** du dataset ?")
                    st.caption("Déplacée vers `dataset_recolte/corbeille/` — récupérable à la main.")
                    if st.button("Confirmer la suppression", key=f"del_{v_name}", use_container_width=True):
                        dataset_tools.move_to_trash(img_dir, lbl_dir, SAVE_DIR / "corbeille", v_name)
                        annotation_timer.log_annotation(
                            "verify", image_name=v_name, source="verify",
                            nb_boxes=0, status="deleted",
                        )
                        st.session_state.pop(f"_verify_pred_{v_name}", None)
                        if idx + 1 < len(names):
                            st.session_state["_verify_jump"] = names[idx + 1]
                        elif idx > 0:
                            st.session_state["_verify_jump"] = names[idx - 1]
                        st.toast(f"{v_name} déplacée vers la corbeille", icon="🗑")
                        st.rerun()

                img_path = img_dir / v_name
                v_raw_bytes = img_path.read_bytes()
                nparr = np.frombuffer(v_raw_bytes, np.uint8)
                img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    st.error(f"Impossible de décoder l'image `{v_name}`.")
                else:
                    # Éditeur sur version réduite (labels normalisés → coordonnées
                    # exactes) ; le fichier original du dataset n'est pas retouché.
                    img_bgr = helper.resize_keep_ratio(img_bgr, MAX_EDIT_SIDE, MAX_EDIT_SIDE)
                    h_img, w_img = img_bgr.shape[:2]
                    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).convert("RGB")

                    # (Re)démarre le chrono à chaque changement d'image affichée
                    if st.session_state.get("_verify_current") != v_name:
                        st.session_state["_verify_current"] = v_name
                        annotation_timer.start_timer("verify")

                    lbl_path = lbl_dir / f"{Path(v_name).stem}.txt"
                    pred_key = f"_verify_pred_{v_name}"
                    if lbl_path.exists():
                        bboxes, labels_idx = _parse_yolo_label(
                            lbl_path.read_text(encoding="utf-8"), w_img, h_img
                        )
                    elif pred_key in st.session_state:
                        bboxes, labels_idx = st.session_state[pred_key]
                    else:
                        bboxes, labels_idx = [], []
                        if st.button("🤖 Pré-annoter avec le modèle", key=f"pred_btn_{v_name}"):
                            with st.spinner("Analyse par le modèle..."):
                                res = model.predict(img_bgr, conf=PRED_CONF)
                            st.session_state[pred_key] = helper.get_detection_initial_data(res[0], CONF_BY_ID)
                            st.rerun()

                    v_token = st.session_state.get("_verify_token", 0)
                    v_result = st_detection(
                        image=pil_img,
                        label_list=LABEL_LIST,
                        bboxes=bboxes,
                        labels=labels_idx,
                        height=h_img,
                        width=w_img,
                        line_width=2,
                        use_space=False,
                        color_map=helper.CLASS_COLORS,
                        token=v_token,
                        image_name=v_name,
                        start_time_ms=annotation_timer.get_start_time_ms("verify"),
                        key=f"verify_ds_{v_name}_{v_token}_{int(pred_key in st.session_state)}",
                    )

                    if v_result is not None:
                        _save_annotation(
                            pil_img, v_result, w_img, h_img,
                            original_name=v_name, raw_bytes=v_raw_bytes, overwrite=True,
                        )
                        log_row = annotation_timer.log_annotation(
                            "verify", image_name=v_name, source="verify", nb_boxes=len(v_result)
                        )
                        st.session_state.pop(pred_key, None)
                        st.toast(f"Annotation sauvegardée ! (⏱ {log_row['duration_display']})", icon="✅")
                        st.session_state["_verify_token"] = v_token + 1
                        if auto_advance and idx + 1 < len(names):
                            st.session_state["_verify_jump"] = names[idx + 1]
                        st.rerun()

    # ── Mode 2 : import manuel d'une paire image + label (secours) ───────────
    else:
        st.write("Importez une image et son fichier d'annotation YOLO pour vérifier ou corriger les bboxes.")

        col_v1, col_v2 = st.columns(2)
        with col_v1:
            v_img = st.file_uploader("Image", type=["jpg", "jpeg", "png"], key="v_img")
        with col_v2:
            v_lbl = st.file_uploader("Annotation YOLO (.txt)", type=["txt"], key="v_lbl")

        if v_img is not None and v_lbl is not None:
            v_raw_bytes = v_img.getvalue()
            nparr = np.frombuffer(v_raw_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                st.error("Impossible de décoder l'image.")
            else:
                img_bgr = helper.resize_keep_ratio(img_bgr, MAX_EDIT_SIDE, MAX_EDIT_SIDE)
                h_img, w_img = img_bgr.shape[:2]
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb).convert("RGB")

                # Démarre le chrono uniquement quand une NOUVELLE paire image/label est importée
                v_pair_id = f"{v_img.name}:{v_img.size}:{v_lbl.name}:{v_lbl.size}"
                if st.session_state.get("_verify_pair_id") != v_pair_id:
                    st.session_state["_verify_pair_id"] = v_pair_id
                    annotation_timer.start_timer("verify")

                bboxes, labels_idx = _parse_yolo_label(
                    v_lbl.getvalue().decode("utf-8"), w_img, h_img
                )

                st.caption(
                    f"**{v_img.name}** — {len(bboxes)} annotation(s) chargée(s) · "
                    "Cliquez **Valider** pour sauvegarder les corrections."
                )

                v_token = st.session_state.get("_verify_token", 0)
                v_result = st_detection(
                    image=pil_img,
                    label_list=LABEL_LIST,
                    bboxes=bboxes,
                    labels=labels_idx,
                    height=h_img,
                    width=w_img,
                    line_width=2,
                    use_space=False,
                    color_map=helper.CLASS_COLORS,
                    token=v_token,
                    image_name=v_img.name,
                    start_time_ms=annotation_timer.get_start_time_ms("verify"),
                    key=f"verify_{v_img.name}_{v_token}",
                )

                if v_result is not None:
                    _save_annotation(
                        pil_img, v_result, w_img, h_img,
                        original_name=v_img.name, raw_bytes=v_raw_bytes, overwrite=True,
                    )
                    log_row = annotation_timer.log_annotation("verify", image_name=v_img.name, source="verify", nb_boxes=len(v_result))
                    st.toast(f"Annotation corrigée et sauvegardée ! (⏱ {log_row['duration_display']})", icon="✅")
                    st.session_state["_verify_token"] = v_token + 1
                    st.rerun()

# ─── ONGLET 5 : DATASET — SANTÉ & EXPORT ─────────────────────────────────────
with tab_dataset:
    ds_img_dir = SAVE_DIR / "images"
    ds_lbl_dir = SAVE_DIR / "labels"
    ds_images = dataset_tools.list_images(ds_img_dir)
    dist = dataset_tools.class_distribution(ds_lbl_dir, len(LABEL_LIST))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Images", len(ds_images))
    m2.metric("Annotées", dist["n_labels"])
    m3.metric("Bboxes", dist["total_boxes"])
    m4.metric("Labels vides (fond)", dist["n_empty"])

    # ── Répartition des classes ───────────────────────────────────────────────
    st.markdown("#### Répartition des classes")
    if dist["total_boxes"] == 0:
        st.info("Aucune annotation pour le moment.")
    else:
        max_count = max(dist["counts"]) or 1
        order = sorted(range(len(LABEL_LIST)), key=lambda i: -dist["counts"][i])
        rows = []
        for i in order:
            name, count = LABEL_LIST[i], dist["counts"][i]
            pct = 100 * count / dist["total_boxes"]
            width = 100 * count / max_count
            color = helper.CLASS_COLORS.get(name, "#888")
            bar = (
                f"<div class='dist-bar' style='width:{width:.1f}%;background:{color};'></div>"
                if count else ""
            )
            rows.append(
                f"<div class='dist-row{' dist-zero' if not count else ''}'>"
                f"<span class='dist-name'><span class='dist-swatch' style='background:{color};'></span>{name}</span>"
                f"<div class='dist-track'>{bar}</div>"
                f"<span class='dist-count'>{count}&nbsp;·&nbsp;{pct:.0f}&thinsp;%</span>"
                f"</div>"
            )
        st.markdown("".join(rows), unsafe_allow_html=True)

        missing = [LABEL_LIST[i] for i in range(len(LABEL_LIST)) if dist["counts"][i] == 0]
        if missing:
            st.caption(
                f"⚠ Classes du référentiel absentes du dataset : **{', '.join(missing)}** — "
                "le prochain modèle ne saura pas les détecter."
            )

    st.divider()

    # ── Anomalies ─────────────────────────────────────────────────────────────
    st.markdown("#### Santé du dataset")
    if st.button("🔎 Analyser les anomalies"):
        with st.spinner("Analyse (doublons de contenu inclus)…"):
            st.session_state["dataset_report"] = dataset_tools.analyze_dataset(
                ds_img_dir, ds_lbl_dir, len(LABEL_LIST)
            )

    report = st.session_state.get("dataset_report")
    if report is not None:
        if report.n_problems == 0:
            st.success("Aucune anomalie détectée ✔")
        else:
            st.warning(f"{report.n_problems} anomalie(s) à examiner :")
            if report.orphan_labels:
                with st.expander(f"🏷 Labels orphelins (sans image) — {len(report.orphan_labels)}"):
                    st.write("\n".join(f"- `{n}`" for n in report.orphan_labels))
            if report.invalid_lines:
                with st.expander(f"⛔ Lignes illisibles — {len(report.invalid_lines)}"):
                    st.write("\n".join(
                        f"- `{f}` ligne {ln} : `{content}`" for f, ln, content in report.invalid_lines
                    ))
            if report.out_of_range:
                with st.expander(f"🔢 Classes hors référentiel — {len(report.out_of_range)}"):
                    st.write("\n".join(
                        f"- `{f}` ligne {ln} : classe {cid}" for f, ln, cid in report.out_of_range
                    ))
            if report.degenerate_boxes:
                with st.expander(f"📐 Bboxes dégénérées ou hors cadre — {len(report.degenerate_boxes)}"):
                    st.caption("Boxes quasi nulles (« fantômes » de l'ancien bug de clic) ou dépassant "
                               "l'image — à corriger dans l'onglet Vérification.")
                    st.write("\n".join(
                        f"- `{f}` ligne {ln} : {detail}" for f, ln, detail in report.degenerate_boxes
                    ))
            if report.duplicate_groups:
                with st.expander(f"👯 Images au contenu identique — {len(report.duplicate_groups)} groupe(s)"):
                    st.caption("Le même contenu sous plusieurs noms fausse le split train/val "
                               "(la même photo peut se retrouver des deux côtés).")
                    st.write("\n".join(
                        "- " + " = ".join(f"`{n}`" for n in grp) for grp in report.duplicate_groups
                    ))
        if report.unlabeled_images:
            st.caption(f"ℹ {len(report.unlabeled_images)} image(s) sans label — à annoter "
                       "via l'onglet Vérification (filtre « ⬜ Sans annotation »).")

    st.divider()

    # ── Export pour entraînement ──────────────────────────────────────────────
    st.markdown("#### Export pour entraînement")
    n_exportable = len({p.stem for p in ds_images} & set(dataset_tools.label_classes(ds_lbl_dir)))
    st.caption(
        f"**{n_exportable}** paires image + label exportables. Split stratifié : chaque image est "
        "rattachée à sa classe la plus rare pour que les classes peu représentées existent aussi "
        f"en validation. Génère `data.yaml` (les {len(LABEL_LIST)} classes du référentiel, accents inclus)."
    )
    val_pct = st.slider("Part de validation (%)", 10, 40, 20, 5)
    if st.button("📦 Exporter vers exports/", type="primary", disabled=n_exportable == 0):
        try:
            with st.spinner("Copie des images et labels…"):
                st.session_state["last_export"] = dataset_tools.export_dataset(
                    ds_img_dir, ds_lbl_dir, settings.ROOT / "exports",
                    LABEL_LIST, val_ratio=val_pct / 100,
                )
        except ValueError as e:
            st.error(str(e))

    exp = st.session_state.get("last_export")
    if exp is not None:
        rel_yaml = exp["yaml_path"].relative_to(settings.ROOT)
        st.success(
            f"Export terminé : **{exp['n_train']} train / {exp['n_val']} val** → "
            f"`{exp['out_dir'].relative_to(settings.ROOT)}`"
        )
        table = ["| Classe | Images train | Images val |", "|---|---|---|"] + [
            f"| {name} | {tr} | {va} |"
            for name, (tr, va) in exp["per_class"].items() if tr or va
        ]
        st.markdown("\n".join(table))
        st.markdown("**Réentraîner puis déployer :**")
        st.code(
            f"python train.py --data {rel_yaml}\n"
            "# puis copier le best.pt produit (chemin affiché en fin d'entraînement)\n"
            "# vers weights/best.pt pour que l'app et le mobile l'utilisent",
            language="bash",
        )