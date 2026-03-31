from ultralytics import YOLO
import time
import streamlit as st
import cv2
import settings
import threading
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import os

CLASS_COLORS = {
    "Dgrx": "#c23232",
    "Mrisq": "#ff8c00",
    "NonCompost": "#5e80ad",
    "Compost": "#16b939",
}

def sleep_and_clear_success():
    """Efface les messages d'alerte après 3 secondes"""
    time.sleep(3)
    if 'recyclable_placeholder' in st.session_state:
        st.session_state['recyclable_placeholder'].empty()
    if 'non_recyclable_placeholder' in st.session_state:
        st.session_state['non_recyclable_placeholder'].empty()
    if 'matiere_risquee' in st.session_state:
        st.session_state['matiere_risquee_placeholder'].empty()
    if 'dangereux_placeholder' in st.session_state:
        st.session_state['dangereux_placeholder'].empty()

def load_model(model_path):
    model = YOLO(model_path)
    return model

def classify_waste_type(detected_items):
    # utilise les listes de settings.py (Compostable, Non-compostable, Mrisq, Dgrx)
    recyclable_items = set(detected_items) & set(settings.RECYCLABLE)
    non_recyclable_items = set(detected_items) & set(settings.NON_RECYCLABLE)
    matiere_risquee_items = set(detected_items) & set(settings.MATIERE_RISQUEE)
    dangereux_items = set(detected_items) & set(settings.DANGEREUX)
    return recyclable_items, non_recyclable_items, matiere_risquee_items, dangereux_items

def remove_dash_from_class_name(class_name):
    return class_name.replace("_", " ")

def _display_detected_frames(model, st_frame, image):
    image = cv2.resize(image, (640, int(640*(9/16))))

    if 'recyclable_placeholder' not in st.session_state:
        st.session_state['recyclable_placeholder'] = st.sidebar.empty()
    if 'non_recyclable_placeholder' not in st.session_state:
        st.session_state['non_recyclable_placeholder'] = st.sidebar.empty()
    if 'matiere_risquee_placeholder' not in st.session_state:
        st.session_state['matiere_risquee_placeholder'] = st.sidebar.empty()
    if 'dangereux_placeholder' not in st.session_state:
        st.session_state['dangereux_placeholder'] = st.sidebar.empty()

    if 'unique_classes' not in st.session_state:
        st.session_state['unique_classes'] = set()

    # inférence
    res = model.predict(image, conf=0.6)
    names = model.names

    # --- LOGIQUE D'AFFICHAGE DES ALERTES ---
    for result in res:
        new_classes = set([names[int(c)] for c in result.boxes.cls])

        # si de nouveaux objets sont détectés
        if new_classes and new_classes != st.session_state['unique_classes']:
            st.session_state['unique_classes'] = new_classes

            # reset des affichages
            st.session_state['recyclable_placeholder'].empty()
            st.session_state['non_recyclable_placeholder'].empty()
            st.session_state['matiere_risquee_placeholder'].empty()
            st.session_state['dangereux_placeholder'].empty()

            compost_items, non_compost_items, matiere_risquee_items, dangereux_items = classify_waste_type(new_classes)

            if compost_items:
                items_str = "\n- ".join(remove_dash_from_class_name(i) for i in compost_items)
                st.session_state['recyclable_placeholder'].markdown(
                    f"<div class='stRecyclable'><b>Compostable(s) :</b>\n\n- {items_str}</div>",
                    unsafe_allow_html=True
                )

            if non_compost_items:
                items_str = "\n- ".join(remove_dash_from_class_name(i) for i in non_compost_items)
                st.session_state['non_recyclable_placeholder'].markdown(
                    f"<div class='stNonRecyclable'><b>Non-Compostable(s) :</b>\n\n- {items_str}</div>",
                    unsafe_allow_html=True
                )

            if matiere_risquee_items:
                items_str = "\n- ".join(remove_dash_from_class_name(i) for i in matiere_risquee_items)
                st.session_state['matiere_risquee_placeholder'].markdown(
                    f"<div class='stMatiereRisquee'><b>Matiere risquee(s) :</b>\n\n- {items_str}</div>",
                    unsafe_allow_html=True
                )

            if dangereux_items:
                items_str = "\n- ".join(remove_dash_from_class_name(i) for i in dangereux_items)
                st.session_state['dangereux_placeholder'].markdown(
                    f"<div class='stDangereux'><b>Dangereux :</b>\n\n- {items_str}</div>",
                    unsafe_allow_html=True
                )

            # timer pour effacer les alertes
            threading.Thread(target=sleep_and_clear_success).start()

    # affichage de l'image
    res_plotted = res[0].plot()
    st_frame.image(res_plotted, channels="BGR")

    # on stocke l'image brute et les prédictions en session_state pour la capture
    st.session_state['last_raw_image'] = image
    st.session_state['last_res'] = res[0]

def play_webcam(model, auto_capture_interval=0):
    source_webcam = settings.WEBCAM_PATH

    col1, col2, col3 = st.columns(3)
    with col1:
        start_btn = st.button('Lancer la détection')
    with col2:
        capture_btn = st.button('Capturer pour annotation')
    with col3:
        stop_btn = st.button('Arrêter')

    if capture_btn:
        st.session_state['run_detection'] = False
        st.session_state['mode_annotation'] = True
        st.rerun()

    if stop_btn:
        st.session_state['run_detection'] = False
        st.rerun()

    queue_info = st.empty()

    if start_btn or st.session_state.get('run_detection', False):
        st.session_state['run_detection'] = True
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            last_capture_time = time.time()
            while (vid_cap.isOpened() and st.session_state.get('run_detection', False)):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(model, st_frame, image)
                    if auto_capture_interval > 0:
                        now = time.time()
                        count = len(st.session_state.get('capture_queue', []))
                        queue_info.info(f"{count} image(s) en attente d'annotation")
                        if now - last_capture_time >= auto_capture_interval:
                            if 'capture_queue' not in st.session_state:
                                st.session_state['capture_queue'] = []
                            st.session_state['capture_queue'].append({
                                'image': st.session_state['last_raw_image'].copy(),
                                'res': st.session_state['last_res']
                            })
                            last_capture_time = now
                else:
                    break
            vid_cap.release()
        except Exception as e:
            st.sidebar.error(f"Erreur : {e}")

def get_canvas_initial_data(results):
    """Transforme les résultats YOLO en données pour le canvas Streamlit"""
    initial_drawing = {"objects": []}
    if results:
        boxes = results.boxes.xywh.cpu().numpy()  # x_center, y_center, w, h
        clss = results.boxes.cls.cpu().numpy()
        names = results.names

        for i, box in enumerate(boxes):
            x_c, y_c, w, h = box
            # Le canvas utilise (left, top, width, height)
            left = float(x_c - (w / 2))
            top = float(y_c - (h / 2))
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
