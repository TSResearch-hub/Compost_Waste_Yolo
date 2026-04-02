from pathlib import Path
import streamlit as st
import helper
import settings
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas
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
button[aria-label="🔴 Dgrx"],
button[aria-label="🟠 Mrisq"],
button[aria-label="🔵 NonCompost"],
button[aria-label="🟢 Compost"] {
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
COLOR_TO_CLASS = {v: k for k, v in helper.CLASS_COLORS.items()}
CLASS_MAP = {"Dgrx": 0, "Mrisq": 1, "NonCompost": 2, "Compost": 3}
SAVE_DIR = Path("dataset_recolte")

# Emoji de couleur pour identification rapide des boutons de classe
CLASS_EMOJI = {"Dgrx": "🔴", "Mrisq": "🟠", "NonCompost": "🔵", "Compost": "🟢"}

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
    """Supprime tout l'état canvas pour ce canvas_key (appelé lors d'un changement d'image)."""
    for suffix in ("_data", "_clear_counter", "_img_id", "_selected_class", "_selected_obj_idx"):
        st.session_state.pop(f"{canvas_key}{suffix}", None)


def _save_annotation(pil_img, canvas_objects, w_img, h_img):
    """
    Fonction de sauvegarde unifiée. Toujours appelée depuis les objets du canvas
    (qu'ils aient été édités ou non), ce qui garantit la cohérence entre ce que
    l'opérateur VOIT et ce qui est enregistré.
    """
    img_dir, lbl_dir = _prepare_save_dirs()
    timestamp = int(time.time())
    pil_img.save(img_dir / f"cap_{timestamp}.jpg")

    yolo_lines = []
    for obj in canvas_objects:
        if obj.get("type") == "rect":
            left = obj["left"]
            top = obj["top"]
            w_box = obj["width"] * obj.get("scaleX", 1)
            h_box = obj["height"] * obj.get("scaleY", 1)
            x_center = (left + w_box / 2) / w_img
            y_center = (top + h_box / 2) / h_img
            norm_w = w_box / w_img
            norm_h = h_box / h_img
            # La couleur du trait encode la classe (même lors d'éditions mixtes)
            obj_label = COLOR_TO_CLASS.get(obj.get("stroke", ""), "Compost")
            class_id = CLASS_MAP.get(obj_label, 3)
            yolo_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
            )

    with open(lbl_dir / f"cap_{timestamp}.txt", "w") as f:
        f.write("\n".join(yolo_lines))


def _exit_annotation(canvas_key, exit_mode_annotation, offline_mode=False):
    """
    Post-sauvegarde/annulation : enchaîne automatiquement sur l'image suivante.
    - offline_mode=True  → avance dans offline_queue (onglet hors ligne)
    - offline_mode=False → avance dans capture_queue (webcam)
    """
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

def show_annotation_editor(raw_img, last_res, canvas_key, exit_mode_annotation=True, offline_mode=False):
    """
    Éditeur d'annotation YOLO — Mode Focus Absolu.

    Architecture "Zone de Frappe" :
      LIGNE 1 — [Mode radio | Boutons de classe ×4 à 60px]  ← juste au-dessus du canvas
      LIGNE 2 — Canvas centré (640×360)
      LIGNE 3 — [✅ VALIDER — pleine largeur | 🗑 Effacer | ✖ Annuler]  ← juste en dessous

    Tout est dans une colonne centrale pour que les yeux ne bougent pas.
    La sidebar est fermée par défaut (set_page_config) → max espace horizontal.
    display_toolbar=False → barre d'outils native supprimée (corbeille bugguée).
    """
    h_img, w_img = raw_img.shape[:2]
    img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb).convert("RGB")

    # ── Session state — données canvas liées à cette image ───────────────────
    _data_key    = f"{canvas_key}_data"
    _imgid_key   = f"{canvas_key}_img_id"
    _counter_key = f"{canvas_key}_clear_counter"
    _selcls_key  = f"{canvas_key}_selected_class"
    _selidx_key  = f"{canvas_key}_selected_obj_idx"

    if st.session_state.get(_imgid_key) != id(raw_img):
        st.session_state[_imgid_key]    = id(raw_img)
        st.session_state[_data_key]     = helper.get_canvas_initial_data(last_res)
        st.session_state[_counter_key]  = 0
        st.session_state.pop(_selcls_key, None)
        st.session_state.pop(_selidx_key, None)

    initial_data   = st.session_state[_data_key]
    clear_counter  = st.session_state.get(_counter_key, 0)
    selected_class = st.session_state.get(_selcls_key, "Compost")

    # ── Conteneur éditeur ────────────────────────────────────────────────────
    # Pas de colonne de centrage externe : Streamlit limite la nesting à 2 niveaux.
    # Le centrage du canvas est délégué au CSS (.editor-canvas-wrap).
    st.markdown("<div class='editor-container'>", unsafe_allow_html=True)

    # ── LIGNE 1 : Mode + Sélection de classe (niveau 1) ──────────────────────
    # tc_mode / tc_classes = niveau 1. cls_cols à l'intérieur = niveau 2. ✓
    tc_mode, tc_classes = st.columns([2, 5])

    with tc_mode:
        mode = st.radio(
            "Outil",
            ["Modifier/Supprimer", "Dessiner"],
            horizontal=True,
            key=f"mode_{canvas_key}",
        )
        drawing_mode = "transform" if mode == "Modifier/Supprimer" else "rect"

    # 4 boutons de classe à 60px — ciblés via aria-label en CSS.
    # En mode Modifier : le clic reclasse le rectangle sélectionné
    # (Fabric.js bringToFront → dernier élément du JSON = objet actif).
    with tc_classes:
        class_clicked = None
        cls_cols = st.columns(len(helper.CLASS_COLORS))  # niveau 2 ✓
        for col, cn in zip(cls_cols, helper.CLASS_COLORS):
            with col:
                if st.button(
                    f"{CLASS_EMOJI[cn]} {cn}",
                    key=f"cls_{canvas_key}_{cn}",
                    type="primary" if cn == selected_class else "secondary",
                    use_container_width=True,
                ):
                    class_clicked = cn
                    selected_class = cn
                    st.session_state[_selcls_key] = cn

    stroke_color = helper.CLASS_COLORS.get(selected_class, "#ff0000")

    # ── LIGNE 2 : Canvas (centré via CSS) ────────────────────────────────────
    # display_toolbar=False : supprime la corbeille native qui réappliquait
    # les prédictions IA au lieu de vraiment vider le canvas.
    st.markdown("<div class='editor-canvas-wrap'>", unsafe_allow_html=True)
    canvas_result = st_canvas(
        initial_drawing=initial_data,
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=2,
        stroke_color=stroke_color,
        background_image=pil_img,
        update_streamlit=True,
        height=h_img,
        width=w_img,
        drawing_mode=drawing_mode,
        display_toolbar=False,
        key=f"{canvas_key}_{clear_counter}",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Objets courants (canvas ou session_state si canvas pas encore rendu) ──
    if canvas_result.json_data and canvas_result.json_data.get("objects"):
        current_objects = canvas_result.json_data["objects"]
    else:
        current_objects = st.session_state.get(_data_key, {}).get("objects", [])

    # ── Reclassement de l'OBJET CIBLÉ (corrige le bug [-1] qui ciblait
    #    toujours le dernier créé, quelle que soit la sélection canvas).
    if class_clicked and drawing_mode == "transform" and current_objects:
        updated  = [dict(o) for o in current_objects]
        sel_idx  = st.session_state.get(_selidx_key, 0)
        sel_idx  = max(0, min(sel_idx, len(updated) - 1))
        updated[sel_idx]["stroke"] = helper.CLASS_COLORS[class_clicked]
        st.session_state[_data_key]    = {"objects": updated}
        st.session_state[_counter_key] = clear_counter + 1
        st.rerun()

    # ── LIGNE 3 : Sélecteur d'objet + Suppression (mode Modifier uniquement) ──
    # Remplace le clic-droit (non capturé par streamlit-drawable-canvas) :
    # boutons numérotés colorés par classe + bouton Supprimer dédié.
    if drawing_mode == "transform" and current_objects:
        n_obj   = len(current_objects)
        sel_idx = st.session_state.get(_selidx_key, 0)
        sel_idx = max(0, min(sel_idx, n_obj - 1))

        lbl_col, *obj_cols, del_col = st.columns([2] + [1] * n_obj + [2])
        with lbl_col:
            st.markdown(
                "<div style='padding-top:0.55rem;font-weight:600;font-size:0.95rem;'>"
                "Objet ciblé :</div>",
                unsafe_allow_html=True,
            )
        for i, (col, obj) in enumerate(zip(obj_cols, current_objects)):
            obj_class = COLOR_TO_CLASS.get(obj.get("stroke", ""), "Compost")
            obj_emoji = CLASS_EMOJI.get(obj_class, "⬜")
            with col:
                if st.button(
                    f"{obj_emoji} {i + 1}",
                    key=f"selidx_{canvas_key}_{i}",
                    type="primary" if i == sel_idx else "secondary",
                    use_container_width=True,
                ):
                    st.session_state[_selidx_key] = i
        with del_col:
            if st.button("🗑 Supprimer", key=f"del_obj_{canvas_key}", use_container_width=True):
                updated = [dict(o) for o in current_objects]
                updated.pop(sel_idx)
                st.session_state[_data_key]    = {"objects": updated}
                st.session_state[_counter_key] = clear_counter + 1
                if updated:
                    st.session_state[_selidx_key] = min(sel_idx, len(updated) - 1)
                else:
                    st.session_state.pop(_selidx_key, None)
                st.rerun()

    # ── LIGNE 4 : Actions ─────────────────────────────────────────────────────
    st.write("")
    va_validate, va_clear, va_cancel = st.columns([6, 1, 1])

    with va_validate:
        if st.button(
            "✅  Valider l'annotation",
            key=f"validate_{canvas_key}",
            type="primary",
            use_container_width=True,
        ):
            if canvas_result.json_data is not None:
                objects = canvas_result.json_data.get("objects", [])
            else:
                objects = helper.get_canvas_initial_data(last_res).get("objects", [])
            _save_annotation(pil_img, objects, w_img, h_img)
            st.toast("Annotation sauvegardée !", icon="✅")
            _exit_annotation(canvas_key, exit_mode_annotation, offline_mode=offline_mode)

    with va_clear:
        if st.button("🗑 Effacer", key=f"clear_{canvas_key}", use_container_width=True):
            st.session_state[_data_key]    = {"objects": []}
            st.session_state[_counter_key] = clear_counter + 1
            st.rerun()

    with va_cancel:
        if st.button("✖ Annuler", key=f"cancel_{canvas_key}", use_container_width=True):
            if exit_mode_annotation:
                _exit_annotation(canvas_key, exit_mode_annotation, offline_mode=offline_mode)

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ONGLETS PRINCIPAUX
# ══════════════════════════════════════════════════════════════════════════════
tab_detection, tab_offline = st.tabs(["📹 Détection en direct", "🗂 Annotation hors ligne"])

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
                    nparr = np.frombuffer(f.read(), np.uint8)
                    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    img_resized = cv2.resize(img_bgr, (640, int(640 * (9 / 16))))
                    items.append({"image": img_resized, "name": f.name, "res": None})
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
        )

        # Nettoyage du compteur total quand la file est épuisée après rerun
        if not st.session_state.get("offline_queue"):
            st.session_state.pop("offline_queue_total", None)