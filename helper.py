from ultralytics import YOLO
import time
import streamlit as st
import cv2
import settings
from PIL import Image

CLASS_COLORS = {
    "Dgrx": "#c23232",
    "Mrisq": "#ff8c00",
    "NonCompost": "#5e80ad",
    "Compost": "#16b939",
}


def load_model(model_path):
    return YOLO(model_path)


def classify_waste_type(detected_items):
    """Catégorise les éléments détectés selon les listes de settings.py."""
    recyclable_items = set(detected_items) & set(settings.RECYCLABLE)
    non_recyclable_items = set(detected_items) & set(settings.NON_RECYCLABLE)
    matiere_risquee_items = set(detected_items) & set(settings.MATIERE_RISQUEE)
    dangereux_items = set(detected_items) & set(settings.DANGEREUX)
    return recyclable_items, non_recyclable_items, matiere_risquee_items, dangereux_items


def remove_dash_from_class_name(class_name):
    return class_name.replace("_", " ")


def render_alert_panel(compost_items, non_compost_items, risk_items, danger_items):
    """
    Génère le HTML du panneau d'alertes "feu tricolore industriel".
    Ordre d'affichage : Danger (priorité max) → Risqué → Non-compostable → Compostable.
    Les blocs sont larges et très lisibles pour un usage avec gants sur écran tactile.
    """
    blocks = []

    if danger_items:
        items_str = ", ".join(remove_dash_from_class_name(i) for i in danger_items)
        blocks.append(
            f"<div class='alert-block alert-danger'>"
            f"<span class='alert-icon'>☠</span>"
            f"<div><strong>DANGEREUX</strong><small>{items_str}</small></div>"
            f"</div>"
        )

    if risk_items:
        items_str = ", ".join(remove_dash_from_class_name(i) for i in risk_items)
        blocks.append(
            f"<div class='alert-block alert-risk'>"
            f"<span class='alert-icon'>⚠</span>"
            f"<div><strong>RISQUÉ</strong><small>{items_str}</small></div>"
            f"</div>"
        )

    if non_compost_items:
        items_str = ", ".join(remove_dash_from_class_name(i) for i in non_compost_items)
        blocks.append(
            f"<div class='alert-block alert-non-compost'>"
            f"<span class='alert-icon'>✖</span>"
            f"<div><strong>NON-COMPOSTABLE</strong><small>{items_str}</small></div>"
            f"</div>"
        )

    if compost_items:
        items_str = ", ".join(remove_dash_from_class_name(i) for i in compost_items)
        blocks.append(
            f"<div class='alert-block alert-compost'>"
            f"<span class='alert-icon'>✔</span>"
            f"<div><strong>COMPOSTABLE</strong><small>{items_str}</small></div>"
            f"</div>"
        )

    return "".join(blocks)


def _display_detected_frames(model, st_frame, image, conf=0.4):
    """
    Effectue l'inférence YOLO sur une frame et met à jour le placeholder vidéo.
    Retourne l'ensemble des classes détectées pour que l'appelant gère les alertes.
    """
    image = cv2.resize(image, (640, int(640 * (9 / 16))))

    res = model.predict(image, conf=conf)
    names = model.names

    # Collecte toutes les classes détectées sur la frame
    detected_classes = set()
    for result in res:
        detected_classes = {names[int(c)] for c in result.boxes.cls}

    res_plotted = res[0].plot()
    st_frame.image(res_plotted, channels="BGR")

    # Stockage en session pour la capture et l'annotation
    st.session_state['last_raw_image'] = image
    st.session_state['last_res'] = res[0]

    return detected_classes


def play_webcam(model, alert_placeholder, auto_capture_interval=0, conf=0.4, camera_index=None):
    """
    Boucle de capture webcam avec détection YOLO en temps réel.

    Ergonomie industrielle :
    - Boutons larges et espacés, directement sous le flux vidéo
    - Alertes mises à jour dans `alert_placeholder` (panneau latéral défini dans app.py)
    - Disparition des alertes gérée par timestamp en session_state (sans thread)
    """
    source_webcam = camera_index if camera_index is not None else settings.WEBCAM_PATH

    # ── Boutons de contrôle principaux ────────────────────────────────────────
    # Placés ici pour rester visuellement solidaires du flux vidéo
    col1, col2, col3 = st.columns([3, 3, 2])
    with col1:
        start_btn = st.button("▶  Lancer la détection", type="primary", use_container_width=True)
    with col2:
        capture_btn = st.button("📷  Capturer pour annotation", use_container_width=True)
    with col3:
        stop_btn = st.button("⏹  Arrêter", use_container_width=True)

    if capture_btn:
        st.session_state['run_detection'] = False
        st.session_state['mode_annotation'] = True
        st.rerun()

    if stop_btn:
        st.session_state['run_detection'] = False
        st.rerun()

    if start_btn or st.session_state.get('run_detection', False):
        st.session_state['run_detection'] = True

        if 'unique_classes' not in st.session_state:
            st.session_state['unique_classes'] = set()

        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            last_capture_time = time.time()

            while vid_cap.isOpened() and st.session_state.get('run_detection', False):
                success, image = vid_cap.read()
                if not success:
                    break

                new_classes = _display_detected_frames(model, st_frame, image, conf=conf)

                # ── Mise à jour du panneau d'alertes (sans threading) ─────────
                # Stratégie : on rafraîchit un timeout en session_state à chaque
                # frame avec détection. Si plus rien n'est détecté ET que le
                # timeout est dépassé, on efface le panneau.
                if new_classes:
                    st.session_state['alert_timeout'] = time.time() + 3.0
                    if new_classes != st.session_state.get('unique_classes', set()):
                        st.session_state['unique_classes'] = new_classes

                    compost, non_compost, risk, danger = classify_waste_type(new_classes)
                    html = render_alert_panel(compost, non_compost, risk, danger)
                    if html:
                        alert_placeholder.markdown(html, unsafe_allow_html=True)
                else:
                    if time.time() > st.session_state.get('alert_timeout', 0):
                        alert_placeholder.empty()
                        st.session_state['unique_classes'] = set()

                # ── Auto-capture ──────────────────────────────────────────────
                if auto_capture_interval > 0:
                    now = time.time()
                    if now - last_capture_time >= auto_capture_interval:
                        if 'capture_queue' not in st.session_state:
                            st.session_state['capture_queue'] = []
                        st.session_state['capture_queue'].append({
                            'image': st.session_state['last_raw_image'].copy(),
                            'res': st.session_state['last_res']
                        })
                        last_capture_time = now

            vid_cap.release()

        except Exception as e:
            st.error(f"Erreur caméra : {e}")


def get_canvas_initial_data(results):
    """Transforme les résultats YOLO (xywh) en objets pour le canvas streamlit-drawable-canvas."""
    initial_drawing = {"objects": []}
    if results:
        boxes = results.boxes.xywh.cpu().numpy()  # x_center, y_center, w, h
        clss = results.boxes.cls.cpu().numpy()
        names = results.names

        for i, box in enumerate(boxes):
            x_c, y_c, w, h = box
            # Conversion : centre + taille → coin haut-gauche + taille (format canvas)
            left = float(x_c - w / 2)
            top = float(y_c - h / 2)
            class_name = names[int(clss[i])]

            initial_drawing["objects"].append({
                "type": "rect",
                "left": left,
                "top": top,
                "width": float(w),
                "height": float(h),
                "fill": "rgba(255, 165, 0, 0.3)",
                "stroke": CLASS_COLORS.get(class_name, "#ff0000"),
                "strokeWidth": 2,
            })
    return initial_drawing