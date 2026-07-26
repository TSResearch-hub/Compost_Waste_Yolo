# Image unique du projet : les trois interfaces partagent le même code et les
# mêmes dépendances — seul le point d'entrée change (voir docker-compose.yml).
#
# L'image ne contient QUE le code (voir .dockerignore) : données, modèles et
# sorties vivent sur la machine hôte et sont montés en volumes par compose.
FROM python:3.12-slim

# Bibliothèques système requises par opencv-python sur une image "slim"
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch CPU AVANT le reste : ~700 Mo au lieu de ~4 Go en version CUDA.
# Les conteneurs font de l'inférence (annotation, surveillance) — les
# entraînements restent sur Colab ou sur la machine hôte avec GPU.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Dépendances d'abord (couche mise en cache tant que requirements/pyproject ne
# changent pas), code ensuite : modifier le code ne réinstalle pas les paquets.
COPY requirements.txt pyproject.toml ./
COPY src/ src/
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV MPLBACKEND=Agg \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# 8501 annotation · 8502 production · 8000 mobile
EXPOSE 8501 8502 8000
