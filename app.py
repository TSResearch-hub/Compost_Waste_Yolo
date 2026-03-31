from pathlib import Path
import streamlit as st
import helper
import settings
import cv2
from PIL import Image
import os
from streamlit_drawable_canvas import st_canvas
import time
import numpy as np

st.set_page_config(page_title="Composte IA")

st.sidebar.title("Console de détection")

model_path = Path(settings.DETECTION_MODEL)

st.title("Classification automatisée de déchets pour station de compostage")
st.markdown(
"""
<style>
    .stRecyclable {
        background-color: rgba(233,192,78,255);
        padding: 1rem 0.75rem;
        margin-bottom: 1rem;
        border-radius: 0.5rem;
        margin-top: 0 !important;
        font-size:18px !important;
    }
    .stNonRecyclable {
        background-color: rgba(94,128,173,255);
        padding: 1rem 0.75rem;
        margin-bottom: 1rem;
        border-radius: 0.5rem;
        margin-top: 0 !important;
        font-size:18px !important;
    }
    .stMatiereRisquee {
        background-color: rgba(194,84,85,255);
        padding: 1rem 0.75rem;
        margin-bottom: 1rem;
        border-radius: 0.5rem;
        margin-top: 0 !important;
        font-size:18px !important;
    }
    .stDangereux {
        background-color: rgba(194,84,85,255);
        padding: 1rem 0.75rem;
        margin-bottom: 1rem;
        border-radius: 0.5rem;
        margin-top: 0 !important;
        font-size:18px !important;
    }
</style>
""",
unsafe_allow_html=True
)

try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)
    st.stop()

# Correspondance couleur → classe pour l'export (fix bug)
COLOR_TO_CLASS = {v: k for k, v in helper.CLASS_COLORS.items()}
CLASS_MAP = {"Dgrx": 0, "Mrisq": 1, "NonCompost": 2, "Compost": 3}
SAVE_DIR = Path("dataset_recolte")


def show_annotation_editor(raw_img, last_res, canvas_key, exit_mode_annotation=True):
    """Affiche l'éditeur d'annotation YOLO avec canvas interactif."""
    st.divider()
    st.subheader("Éditeur d'Annotation YOLO")

    h_img, w_img = raw_img.shape[:2]
    img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb).convert("RGB")

    col_mode, col_class = st.columns(2)
    with col_mode:
        mode = st.radio(
            "Outil :",
            ["Modifier/Supprimer", "Dessiner un nouveau"],
            horizontal=True,
            key=f"mode_{canvas_key}"
        )
        drawing_mode = "transform" if mode == "Modifier/Supprimer" else "rect"
    with col_class:
        selected_class = st.selectbox(
            "Classe pour les nouveaux rectangles",
            ["Dgrx", "Mrisq", "NonCompost", "Compost"],
            index=3,
            key=f"class_{canvas_key}"
        )

    # Légende des couleurs
    legend_html = " &nbsp; ".join(
        f"<span style='background:{c};padding:2px 8px;border-radius:4px;font-size:13px'>{n}</span>"
        for n, c in helper.CLASS_COLORS.items()
    )
    st.markdown(legend_html, unsafe_allow_html=True)

    initial_data = helper.get_canvas_initial_data(last_res)
    stroke_color = helper.CLASS_COLORS.get(selected_class, "#ff0000")

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
        key=canvas_key,
    )

    col_validate, col_save, col_cancel = st.columns(3)

    with col_validate:
        if st.button("Valider tel quel", key=f"validate_{canvas_key}"):
            # Sauvegarde directe depuis les résultats du modèle, sans édition canvas
            img_dir = SAVE_DIR / "images"
            lbl_dir = SAVE_DIR / "labels"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time())
            pil_img.save(img_dir / f"cap_{timestamp}.jpg")

            yolo_lines = []
            if last_res and len(last_res.boxes) > 0:
                boxes = last_res.boxes.xywh.cpu().numpy()
                clss = last_res.boxes.cls.cpu().numpy()
                names = last_res.names
                for i, box in enumerate(boxes):
                    x_c, y_c, w, h = box
                    class_id = CLASS_MAP.get(names[int(clss[i])], 3)
                    yolo_lines.append(
                        f"{class_id} {x_c/w_img:.6f} {y_c/h_img:.6f} {w/w_img:.6f} {h/h_img:.6f}"
                    )

            with open(lbl_dir / f"cap_{timestamp}.txt", "w") as f:
                f.write("\n".join(yolo_lines))

            st.success(f"Annotation sauvegardée dans {SAVE_DIR} !")
            time.sleep(1)
            if exit_mode_annotation:
                st.session_state['mode_annotation'] = False
                st.rerun()

    with col_save:
        if st.button("Enregistrer les modifications", key=f"save_{canvas_key}"):
            if canvas_result.json_data is not None:
                img_dir = SAVE_DIR / "images"
                lbl_dir = SAVE_DIR / "labels"
                img_dir.mkdir(parents=True, exist_ok=True)
                lbl_dir.mkdir(parents=True, exist_ok=True)
                timestamp = int(time.time())
                pil_img.save(img_dir / f"cap_{timestamp}.jpg")

                yolo_lines = []
                for obj in canvas_result.json_data["objects"]:
                    if obj["type"] == "rect":
                        left, top = obj["left"], obj["top"]
                        w_box = obj["width"] * obj["scaleX"]
                        h_box = obj["height"] * obj["scaleY"]
                        x_center = (left + (w_box / 2)) / w_img
                        y_center = (top + (h_box / 2)) / h_img
                        norm_w = w_box / w_img
                        norm_h = h_box / h_img

                        # Fix : on lit la couleur du trait pour retrouver la classe
                        obj_label = COLOR_TO_CLASS.get(obj.get("stroke", ""), selected_class)
                        class_id = CLASS_MAP.get(obj_label, 3)
                        yolo_lines.append(
                            f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
                        )

                with open(lbl_dir / f"cap_{timestamp}.txt", "w") as f:
                    f.write("\n".join(yolo_lines))

                st.success(f"Enregistré dans {SAVE_DIR} !")
                time.sleep(1)
                if exit_mode_annotation:
                    st.session_state['mode_annotation'] = False
                    st.rerun()

    with col_cancel:
        if st.button("Annuler", key=f"cancel_{canvas_key}"):
            if exit_mode_annotation:
                st.session_state['mode_annotation'] = False
                st.rerun()


# --- ONGLETS PRINCIPAUX ---
tab_detection, tab_offline = st.tabs(["Détection en direct", "Annotation hors ligne"])

with tab_detection:
    st.write("Lancez la détection sur le flux webcam.")

    col_opt1, col_opt2 = st.columns([1, 2])
    with col_opt1:
        auto_capture_on = st.checkbox("Capture automatique")
    with col_opt2:
        if auto_capture_on:
            auto_interval = st.slider("Intervalle (secondes)", min_value=1, max_value=60, value=5)
        else:
            auto_interval = 0

    st.sidebar.markdown(".", unsafe_allow_html=True)

    helper.play_webcam(model, auto_capture_interval=auto_interval)

    # File d'attente des captures automatiques
    queue = st.session_state.get('capture_queue', [])
    if queue:
        st.info(f"{len(queue)} image(s) en attente d'annotation.")
        col_next, col_clear = st.columns(2)
        with col_next:
            if st.button("Annoter la prochaine image"):
                next_item = st.session_state['capture_queue'].pop(0)
                st.session_state['last_raw_image'] = next_item['image']
                st.session_state['last_res'] = next_item['res']
                st.session_state['mode_annotation'] = True
                st.rerun()
        with col_clear:
            if st.button("Vider la file"):
                st.session_state['capture_queue'] = []
                st.rerun()

    # Éditeur d'annotation (webcam)
    if st.session_state.get('mode_annotation', False) and 'last_raw_image' in st.session_state:
        time.sleep(0.1)
        show_annotation_editor(
            st.session_state['last_raw_image'],
            st.session_state.get('last_res'),
            canvas_key="canvas_webcam",
            exit_mode_annotation=True
        )

with tab_offline:
    st.write("Importez une image pour l'annoter manuellement.")
    uploaded_file = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img_bytes = uploaded_file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_resized = cv2.resize(img_bgr, (640, int(640 * (9 / 16))))

        with st.spinner("Analyse par le modèle..."):
            res = model.predict(img_resized, conf=0.6)

        time.sleep(0.1)
        show_annotation_editor(
            img_resized,
            res[0],
            canvas_key="canvas_offline",
            exit_mode_annotation=False
        )
