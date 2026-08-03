"""LED d'alerte et bouton de capture sur les broches GPIO (Jetson, Raspberry).

Le câblage (numéros de broches BOARD du connecteur 40 broches) est décrit dans
configs/hardware.yaml. Sur une machine sans GPIO (PC, WSL, Docker), le module
se désactive silencieusement : l'interface fonctionne à l'identique, sans LED
ni bouton physique.

La LED reflète l'état d'alerte (allumée tant qu'un intrus est signalé) ; le
bouton déclenche une capture manuelle, comme le bouton « Capture manuelle »
de l'interface — pour un intrus que le modèle ne voit pas.
"""

import time
from pathlib import Path

import yaml

DEFAULT_LED_PIN = 12
DEFAULT_BUTTON_PIN = 18


def _import_gpio():
    """Jetson.GPIO et RPi.GPIO partagent la même API ; None si aucun des deux."""
    try:
        import Jetson.GPIO as gpio
        return gpio, None
    except ImportError:
        pass
    try:
        import RPi.GPIO as gpio
        return gpio, None
    except ImportError:
        return None, "Jetson.GPIO / RPi.GPIO non installé"


class HardwareIO:
    """LED (sortie) + bouton poussoir (entrée) ; sans GPIO, tout est no-op.

    Le bouton est câblé vers le 3,3 V : appuyé = niveau haut. La lecture est
    scrutée à chaque frame de la boucle de surveillance (pas d'interruption),
    avec anti-rebond logiciel : ``button_pressed()`` ne rend True qu'une fois
    par appui.
    """

    def __init__(self, led_pin=DEFAULT_LED_PIN, button_pin=DEFAULT_BUTTON_PIN,
                 gpio=None, debounce_s=0.3):
        self.led_pin, self.button_pin, self.debounce_s = led_pin, button_pin, debounce_s
        self.reason = None
        if gpio is None:
            gpio, self.reason = _import_gpio()
        self.gpio = gpio
        self._led_on = False
        self._was_pressed = False
        self._last_press = 0.0
        if self.gpio is not None:
            try:
                self.gpio.setmode(self.gpio.BOARD)
                if self.led_pin:
                    self.gpio.setup(self.led_pin, self.gpio.OUT,
                                    initial=self.gpio.LOW)
                if self.button_pin:
                    self.gpio.setup(self.button_pin, self.gpio.IN)
            except Exception as e:  # broche invalide, droits insuffisants...
                self.gpio, self.reason = None, f"initialisation GPIO impossible : {e}"

    @classmethod
    def from_config(cls, path):
        """Broches lues dans le yaml ; défauts du module si le fichier manque."""
        cfg = {}
        if Path(path).exists():
            cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(led_pin=cfg.get("led_pin", DEFAULT_LED_PIN),
                   button_pin=cfg.get("button_pin", DEFAULT_BUTTON_PIN))

    @property
    def enabled(self):
        return self.gpio is not None

    def set_alert(self, on):
        """Allume/éteint la LED ; n'écrit sur la broche qu'aux changements."""
        if self.gpio is None or not self.led_pin or on == self._led_on:
            return
        self.gpio.output(self.led_pin,
                         self.gpio.HIGH if on else self.gpio.LOW)
        self._led_on = on

    def button_pressed(self):
        """True une seule fois par appui (front montant + anti-rebond)."""
        if self.gpio is None or not self.button_pin:
            return False
        pressed_now = self.gpio.input(self.button_pin) == self.gpio.HIGH
        fired = (pressed_now and not self._was_pressed
                 and time.monotonic() - self._last_press >= self.debounce_s)
        self._was_pressed = pressed_now
        if fired:
            self._last_press = time.monotonic()
        return fired

    def close(self):
        """Éteint la LED et rend les broches (les autres restent intactes)."""
        if self.gpio is None:
            return
        try:
            self.set_alert(False)
            self.gpio.cleanup([p for p in (self.led_pin, self.button_pin) if p])
        except Exception:
            pass
        self.gpio = None
