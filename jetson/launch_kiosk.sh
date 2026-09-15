#!/bin/bash
# Attendre quelques secondes que le serveur graphique et les périphériques soient prêts
sleep 3

# Se déplacer dans le dossier
cd /home/jetson/Documents/Compost

# Lancer directement le Python de l'environnement virtuel
/home/jetson/Documents/Compost/venv/bin/python app_kiosk.py
