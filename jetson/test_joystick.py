import os
os.environ["SDL_VIDEODRIVER"] = "dummy" # Évite l'erreur d'absence d'écran graphique
import pygame
import time

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("Aucun joystick détecté.")
    exit()

joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"En écoute de l'encodeur : {joystick.get_name()}")

try:
    while True:
        pygame.event.pump() # Met à jour l'état de l'encodeur
        for i in range(joystick.get_numbuttons()):
            if joystick.get_button(i):
                print(f"Action : Bouton {i} activé")
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nFin du test.")
