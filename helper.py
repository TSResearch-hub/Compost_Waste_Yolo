from ultralytics import YOLO
import time
import unicodedata
import streamlit as st
import cv2
import settings
from PIL import Image

CLASS_COLORS = {
    "Plastique":  "#1E88E5",
    "Métal":      "#757575",
    "Carton":     "#795548",
    "Aluminium":  "#90A4AE",
    "Céramique":  "#FF7043",
    "Verre":      "#00897B",
    "Composite":  "#8E24AA",
    "Éponge":     "#FDD835",
}


def load_model(model_path):
    return YOLO(model_path)


def normalize_class_name(name: str) -> str:
    """Minuscules + sans accents : le modèle actuel nomme ses classes sans
    accents (Metal, Ceramique) alors que le référentiel du projet les écrit
    avec (Métal, Céramique). Toute comparaison de noms passe par ici."""
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def classify_waste_type(detected_items):
    """
    Répartit les classes détectées entre les 4 niveaux d'alerte de settings.py.
    Retourne (compostables, non_compostables, risqués, dangereux).
    """
    def _match(reference):
        wanted = {normalize_class_name(r) for r in reference}
        return {item for item in detected_items if normalize_class_name(item) in wanted}

    return (
        _match(settings.COMPOSTABLE),
        _match(settings.NON_COMPOSTABLE),
        _match(settings.MATIERE_RISQUEE),
        _match(settings.DANGEREUX),
    )


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
    # Ratio conservé : forcer 640×360 (16:9) déformait les caméras 4:3 — et
    # c'est cette frame qui part dans le dataset via la capture d'annotation.
    image = resize_keep_ratio(image, max_w=640, max_h=640)

    res = model.predict(image, conf=conf)
    names = model.names
    detected_classes = {names[int(c)] for c in res[0].boxes.cls}

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


# Référentiel aligné sur weights/best.pt (mêmes ids, même ordre que son
# data.yaml — seule différence : les accents, absorbés par normalize_class_name).
# Si un futur modèle change encore d'ordre ou de classes, migrer les labels
# existants avec migrate_labels_8_classes.py comme modèle.
CLASS_MAP = {
    "Plastique": 0,
    "Métal":     1,
    "Carton":    2,
    "Aluminium": 3,
    "Céramique": 4,
    "Verre":     5,
    "Composite": 6,
    "Éponge":    7,
}

_NORM_NAME_TO_ID = {normalize_class_name(name): cid for name, cid in CLASS_MAP.items()}
_warned_unknown_classes: set[str] = set()


def class_name_to_id(class_name: str):
    """Id canonique d'un nom de classe, tolérant accents/casse (Metal → Métal).
    Retourne None si le nom n'existe pas dans le référentiel CLASS_MAP."""
    return _NORM_NAME_TO_ID.get(normalize_class_name(class_name))


def get_detection_initial_data(results):
    """Transforme les résultats YOLO en bboxes [x, y, w, h] (coin haut-gauche) pour detection()."""
    bboxes, labels = [], []
    if not results:
        return bboxes, labels
    boxes = results.boxes.xywh.cpu().numpy()
    clss = results.boxes.cls.cpu().numpy()
    names = results.names
    for i, box in enumerate(boxes):
        x_c, y_c, w, h = box
        bboxes.append([float(x_c - w / 2), float(y_c - h / 2), float(w), float(h)])
        class_name = names[int(clss[i])]
        cid = class_name_to_id(class_name)
        if cid is None:
            # Classe du modèle absente du référentiel : visible en console
            # plutôt que classée silencieusement au mauvais endroit.
            if class_name not in _warned_unknown_classes:
                _warned_unknown_classes.add(class_name)
                print(f"[helper] Classe modèle inconnue du référentiel : {class_name!r} → {list(CLASS_MAP)[0]}")
            cid = 0
        labels.append(cid)
    return bboxes, labels


def resize_keep_ratio(img_bgr, max_w: int = 1280, max_h: int = 720):
    """Redimensionne en conservant le ratio, sans jamais agrandir ni dépasser max_w × max_h."""
    h, w = img_bgr.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale == 1.0:
        return img_bgr
    return cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def extract_frames_from_video(video_path: str, interval_seconds: float) -> list:
    """Extrait une frame toutes les interval_seconds secondes depuis une vidéo."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, int(fps * interval_seconds))

    frames = []
    frame_idx = 0
    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = resize_keep_ratio(frame)
        t = frame_idx / fps
        frames.append({
            "image": frame_resized,
            "name": f"frame_{t:.2f}s.jpg",
            "res": None,
        })
        frame_idx += frame_step

    cap.release()
    return frames