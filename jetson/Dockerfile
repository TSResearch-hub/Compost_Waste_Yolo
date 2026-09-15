# syntax=docker/dockerfile:1
# =============================================================================
# Kiosque Compost - image pour NVIDIA Jetson sous JetPack 7.x (Ubuntu 24.04, CUDA 13)
#
# Choix de l'image de base
#   La Jetson de référence tourne sous JetPack 7 : Python 3.12 système et toute la
#   pile IA installée via pip (torch 2.13 cu130, TensorRT 11.2, ultralytics 8.4.115).
#   Au 13/09/2026, NVIDIA ne publie pas d'image l4t-jetpack r39 (JetPack 7) sur NGC,
#   Ultralytics ne publie que des tags jetpack4/5/6 et déconseille explicitement
#   son image latest-nvidia-arm64 pour TensorRT sur Jetson. Les images l4t r35/r36
#   (JetPack 5/6) ne fonctionnent pas sur un hôte JetPack 7.
#   On reproduit donc à l'identique l'environnement validé sur la Jetson de
#   référence (requirements-jetson.txt) au-dessus d'Ubuntu 24.04. Le pilote GPU
#   (libcuda, libnvidia-*) est injecté à l'exécution par le runtime nvidia.
#
# Construction (sur la Jetson) : docker compose build   (~4 Go téléchargés, 10-20 min)
# =============================================================================
ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Injection du pilote GPU de l'hôte par nvidia-container-runtime
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    # OpenCV (backend Qt/xcb) via le socket X11 partagé avec l'hôte
    QT_X11_NO_MITSHM=1 \
    QT_LOGGING_RULES=qt.qpa.fonts.warning=false \
    # pygame/SDL2 : énumération directe de /dev/input (pas de udev dans le conteneur)
    SDL_JOYSTICK_DISABLE_UDEV=1 \
    SDL_VIDEODRIVER=dummy \
    TZ=Europe/Paris

# --- Dépendances système : Python 3.12, bibliothèques X11/xcb pour cv2.imshow (Qt),
#     wmctrl (plein écran, utilisé par l'application), outils de diagnostic
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip \
        ca-certificates tzdata \
        libgl1 libegl1 libglib2.0-0 libgomp1 \
        libsm6 libxext6 libxrender1 libfontconfig1 libfreetype6 libdbus-1-3 \
        libx11-xcb1 libxkbcommon-x11-0 libxcb-cursor0 libxcb-glx0 libxcb-icccm4 \
        libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0 \
        libxcb-sync1 libxcb-util1 libxcb-xfixes0 libxcb-xinerama0 libxcb-xinput0 libxcb-xkb1 \
        wmctrl x11-utils v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# --- Environnement virtuel (Ubuntu 24.04 interdit pip dans le Python système, PEP 668)
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# --- Pile Python identique à la Jetson de référence + pygame / python-dotenv / requests
#     L'index NVIDIA est indispensable : sur pypi.org, tensorrt-cu13-* ne sont que des
#     archives d'amorçage ; les vraies roues aarch64 sont servies par pypi.nvidia.com.
WORKDIR /app
COPY requirements-jetson.txt .
RUN pip install --upgrade pip \
    && pip install --extra-index-url https://pypi.nvidia.com -r requirements-jetson.txt

# --- Contrôles de build : OpenCV doit avoir un backend graphique (sinon cv2.imshow échoue),
#     et les modules du kiosque doivent s'importer (TensorRT est vérifié à l'exécution, avec le GPU)
RUN python -c "import re, sys, cv2; m = re.search(r'(QT|GTK[+]?):[ ]+YES', cv2.getBuildInformation()); print('OpenCV', cv2.__version__, '- GUI:', m.group(0) if m else 'AUCUN'); sys.exit(0 if m else 1)" \
    && python -c "import torch, ultralytics, pygame, dotenv, requests; print('torch', torch.__version__, '| ultralytics', ultralytics.__version__, '| pygame', pygame.version.ver)"

# --- Application (a_annoter, weights et .env sont montés en volumes par docker-compose)
COPY app_kiosk.py detect.py test_camera.py test_joystick.py ./
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && mkdir -p /app/a_annoter /app/weights

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "app_kiosk.py"]
