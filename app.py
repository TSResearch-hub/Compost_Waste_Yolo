from pathlib import Path
import os
import tempfile
import streamlit as st
import helper
import settings
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

/* ══ BOUTONS DE CLASSE — massifs pour sélection rapide sans erreur ════════
   height: 60px garanti via min-height.
   Ciblés via aria-label que Streamlit expose sur chaque bouton.           */
button[aria-label="Plastique"],
button[aria-label="Métal"],
button[aria-label="Carton"],
button[aria-label="Aluminium"],
button[aria-label="Céramique"],
button[aria-label="Organique"],
button[aria-label="Papier"],
button[aria-label="Verre"],
button[aria-label="Composite"] {
    min-height: 60px !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em;
}

/* ══ BOUTONS DESTRUCTIFS — Effacer (tout) + Supprimer (objet ciblé) ══════ */
button[aria-label="🗑 Effacer"],
button[aria-label="🗑 Supprimer"] {
    min-height: 2.2rem !important;
    font-size: 0.82rem !important;
    opacity: 0.75;
    border: 1.5px solid #c23232 !important;
    color: #c23232 !important;
}
button[aria-label="🗑 Effacer"]:hover,
button[aria-label="🗑 Supprimer"]:hover {
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
# SIDEBAR — réglages profonds uniquement
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ Paramètres")
    conf_threshold = st.slider("Seuil de confiance", 0.1, 0.95, 0.4, 0.05)

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
    st.caption("Composte IA · Station de compostage")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════
SAVE_DIR = Path("dataset_recolte")
LABEL_LIST = list(helper.CLASS_MAP.keys())  # ['Dgrx', 'Mrisq', 'NonCompost', 'Compost']

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


def _save_annotation(pil_img, detection_result, w_img, h_img, original_name=None, raw_bytes=None):
    """
    Sauvegarde l'image et le fichier YOLO depuis le résultat du composant detection().
    detection_result : liste de {'bbox': [x, y, w, h], 'label_id': int, 'label': str}
    Les coordonnées bbox sont en pixels dans l'espace image original.
    Si raw_bytes est fourni, les bytes originaux sont écrits tels quels (sans réencodage).
    """
    img_dir, lbl_dir = _prepare_save_dirs()

    if original_name:
        stem = Path(original_name).stem
        ext  = Path(original_name).suffix.lower() or ".jpg"
    else:
        stem = f"cap_{int(time.time())}"
        ext  = ".jpg"

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


def _exit_annotation(canvas_key, exit_mode_annotation, offline_mode=False):
    """
    Post-sauvegarde/annulation : enchaîne automatiquement sur l'image suivante.
    - offline_mode=True  → avance dans offline_queue (onglet hors ligne)
    - offline_mode=False → avance dans capture_queue (webcam)
    """
    st.session_state["_annot_token"] = st.session_state.get("_annot_token", 0) + 1
    _reset_canvas_state(canvas_key)
    if not exit_mode_annotation:
        return  # Cas hors ligne image unique (legacy)

    if offline_mode:
        # Pop l'image courante ; le prochain rendu de tab_offline affichera la suivante
        queue = st.session_state.get("offline_queue", [])
        if queue:
            queue.pop(0)
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


# ══════════════════════════════════════════════════════════════════════════════
# ÉDITEUR D'ANNOTATION
# ══════════════════════════════════════════════════════════════════════════════

def show_annotation_editor(raw_img, last_res, canvas_key, exit_mode_annotation=True, offline_mode=False, original_name=None, raw_bytes=None):
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

    # Réinitialise les bboxes quand l'image change
    if st.session_state.get(_imgid_key) != id(raw_img):
        st.session_state[_imgid_key]   = id(raw_img)
        bboxes, labels = helper.get_detection_initial_data(last_res)
        st.session_state[_data_key]    = {"bboxes": bboxes, "labels": labels}
        st.session_state[_counter_key] = 0

    clear_counter = st.session_state.get(_counter_key, 0)
    annot_token   = st.session_state.get("_annot_token", 0)
    data = st.session_state[_data_key]

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
        key=f"{canvas_key}_{annot_token}_{clear_counter}",
    )

    # ── Sauvegarde déclenchée par "Complete" dans le composant ────────────────
    if result is not None:
        _save_annotation(pil_img, result, w_img, h_img, original_name=original_name, raw_bytes=raw_bytes)
        st.toast("Annotation sauvegardée !", icon="✅")
        _exit_annotation(canvas_key, exit_mode_annotation, offline_mode=offline_mode)

    # ── Boutons complémentaires ───────────────────────────────────────────────
    st.write("")
    col_clear, col_cancel = st.columns([1, 1])

    with col_clear:
        if st.button("🗑 Effacer tout", key=f"clear_{canvas_key}", use_container_width=True):
            st.session_state[_data_key]    = {"bboxes": [], "labels": []}
            st.session_state[_counter_key] = clear_counter + 1
            st.rerun()

    with col_cancel:
        if st.button("✖ Annuler", key=f"cancel_{canvas_key}", use_container_width=True):
            if exit_mode_annotation:
                _exit_annotation(canvas_key, exit_mode_annotation, offline_mode=offline_mode)

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ONGLETS PRINCIPAUX
# ══════════════════════════════════════════════════════════════════════════════
tab_detection, tab_offline, tab_video, tab_verify = st.tabs(["📹 Détection en direct", "🗂 Annotation hors ligne", "🎬 Extraction vidéo", "🔍 Vérification"])

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
        )

# ─── ONGLET 2 : ANNOTATION HORS LIGNE ────────────────────────────────────────
with tab_offline:
    offline_queue = st.session_state.get("offline_queue", [])

    if not offline_queue:
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
                # Décodage + redimensionnement de toutes les images.
                # La prédiction du modèle est faite en lazy (à l'affichage)
                # pour ne pas bloquer l'interface sur un grand lot.
                items = []
                for f in uploaded_files:
                    raw_bytes = f.getvalue()
                    nparr = np.frombuffer(raw_bytes, np.uint8)
                    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    items.append({"image": img_bgr, "name": f.name, "res": None, "raw_bytes": raw_bytes})
                st.session_state["offline_queue"] = items
                _reset_canvas_state("canvas_offline")
                st.rerun()
    else:
        # ── Annotation en série ───────────────────────────────────────────────
        current = offline_queue[0]
        total   = st.session_state.get("offline_queue_total", len(offline_queue))
        done    = total - len(offline_queue)

        # Initialiser le total au premier lancement (ne plus le changer ensuite)
        if "offline_queue_total" not in st.session_state:
            st.session_state["offline_queue_total"] = total

        # Barre de progression + nom du fichier courant
        st.progress(done / total, text=f"Image {done + 1}/{total} — {current['name']}")

        # Prédiction lazy : exécutée une seule fois par image
        if current["res"] is None:
            with st.spinner("Analyse par le modèle..."):
                res = model.predict(current["image"], conf=conf_threshold)
                current["res"] = res[0]
                # La mutation de current (qui est offline_queue[0]) met à jour
                # session_state directement car c'est le même objet en mémoire.

        col_info, col_stop = st.columns([5, 1])
        with col_info:
            if len(offline_queue) > 1:
                st.caption(f"{len(offline_queue) - 1} image(s) suivront automatiquement.")
        with col_stop:
            if st.button("⏹ Tout arrêter", use_container_width=True):
                st.session_state.pop("offline_queue", None)
                st.session_state.pop("offline_queue_total", None)
                _reset_canvas_state("canvas_offline")
                st.rerun()

        show_annotation_editor(
            current["image"],
            current["res"],
            canvas_key="canvas_offline",
            exit_mode_annotation=True,
            offline_mode=True,
            original_name=current.get("name"),
            raw_bytes=current.get("raw_bytes"),
        )

        # Nettoyage du compteur total quand la file est épuisée après rerun
        if not st.session_state.get("offline_queue"):
            st.session_state.pop("offline_queue_total", None)

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
                st.session_state["offline_queue"] = frames
                st.session_state.pop("offline_queue_total", None)
                _reset_canvas_state("canvas_offline")
                st.toast(f"{len(frames)} images extraites — allez dans 'Annotation hors ligne'", icon="🎬")
                st.rerun()
        finally:
            os.unlink(tmp_path)

# ─── ONGLET 4 : VÉRIFICATION D'ANNOTATION ────────────────────────────────────
with tab_verify:
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
            h_img, w_img = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb).convert("RGB")

            # Parsing du fichier YOLO (coordonnées normalisées → pixels)
            bboxes, labels_idx = [], []
            for line in v_lbl.getvalue().decode("utf-8").strip().split("\n"):
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
                key=f"verify_{v_img.name}_{v_token}",
            )

            if v_result is not None:
                _save_annotation(
                    pil_img, v_result, w_img, h_img,
                    original_name=v_img.name, raw_bytes=v_raw_bytes,
                )
                st.toast("Annotation corrigée et sauvegardée !", icon="✅")
                st.session_state["_verify_token"] = v_token + 1
                st.rerun()