"""Tests de la LED d'alerte et du bouton de capture (GPIO simulé)."""

import pytest

from compost_detection.hardware import (DEFAULT_BUTTON_PIN, DEFAULT_LED_PIN,
                                        HardwareIO)


class FakeGPIO:
    """Reproduit le sous-ensemble de l'API Jetson.GPIO/RPi.GPIO utilisé."""
    BOARD, OUT, IN, HIGH, LOW = "board", "out", "in", 1, 0

    def __init__(self):
        self.mode = None
        self.pins = {}          # broche -> (direction, niveau initial)
        self.writes = []        # (broche, niveau), dans l'ordre
        self.level = self.LOW   # niveau lu sur l'entrée (le bouton)
        self.cleaned = None

    def setmode(self, mode):
        self.mode = mode

    def setup(self, pin, direction, initial=None):
        self.pins[pin] = (direction, initial)

    def output(self, pin, value):
        self.writes.append((pin, value))

    def input(self, pin):
        return self.level

    def cleanup(self, pins):
        self.cleaned = pins


@pytest.fixture
def gpio():
    return FakeGPIO()


def test_sans_gpio_tout_est_inactif():
    hw = HardwareIO()  # ni Jetson.GPIO ni RPi.GPIO sur la machine de test
    assert not hw.enabled
    assert hw.reason is not None
    hw.set_alert(True)               # ne doit pas lever
    assert hw.button_pressed() is False
    hw.close()


def test_setup_configure_les_broches(gpio):
    hw = HardwareIO(gpio=gpio)
    assert hw.enabled
    assert gpio.mode == FakeGPIO.BOARD
    assert gpio.pins[DEFAULT_LED_PIN] == (FakeGPIO.OUT, FakeGPIO.LOW)
    assert gpio.pins[DEFAULT_BUTTON_PIN] == (FakeGPIO.IN, None)


def test_setup_en_echec_desactive_sans_lever(gpio):
    def refuse(*a, **k):
        raise RuntimeError("broche occupée")
    gpio.setup = refuse
    hw = HardwareIO(gpio=gpio)
    assert not hw.enabled
    assert "broche occupée" in hw.reason


def test_led_n_ecrit_qu_aux_changements(gpio):
    hw = HardwareIO(gpio=gpio)
    hw.set_alert(True)
    hw.set_alert(True)   # même état : pas de nouvelle écriture
    hw.set_alert(False)
    assert gpio.writes == [(DEFAULT_LED_PIN, FakeGPIO.HIGH),
                           (DEFAULT_LED_PIN, FakeGPIO.LOW)]


def test_bouton_un_seul_declenchement_par_appui(gpio):
    hw = HardwareIO(gpio=gpio, debounce_s=0)
    assert hw.button_pressed() is False   # repos
    gpio.level = FakeGPIO.HIGH            # appui
    assert hw.button_pressed() is True    # front montant
    assert hw.button_pressed() is False   # maintenu : pas de re-déclenchement
    gpio.level = FakeGPIO.LOW             # relâché
    assert hw.button_pressed() is False
    gpio.level = FakeGPIO.HIGH            # nouvel appui
    assert hw.button_pressed() is True


def test_anti_rebond_ignore_les_appuis_rapproches(gpio):
    hw = HardwareIO(gpio=gpio, debounce_s=60)  # très long : le 2e appui tombe dedans
    gpio.level = FakeGPIO.HIGH
    assert hw.button_pressed() is True
    gpio.level = FakeGPIO.LOW
    hw.button_pressed()
    gpio.level = FakeGPIO.HIGH            # rebond juste après l'appui
    assert hw.button_pressed() is False


def test_close_eteint_la_led_et_rend_les_broches(gpio):
    hw = HardwareIO(gpio=gpio)
    hw.set_alert(True)
    hw.close()
    assert gpio.writes[-1] == (DEFAULT_LED_PIN, FakeGPIO.LOW)
    assert sorted(gpio.cleaned) == sorted([DEFAULT_LED_PIN, DEFAULT_BUTTON_PIN])
    assert not hw.enabled
    hw.close()   # idempotent


def test_from_config_lit_le_yaml(tmp_path):
    cfg = tmp_path / "hardware.yaml"
    cfg.write_text("led_pin: 33\nbutton_pin: 31\n", encoding="utf-8")
    hw = HardwareIO.from_config(cfg)
    assert (hw.led_pin, hw.button_pin) == (33, 31)


def test_from_config_defauts_si_fichier_absent(tmp_path):
    hw = HardwareIO.from_config(tmp_path / "absent.yaml")
    assert (hw.led_pin, hw.button_pin) == (DEFAULT_LED_PIN, DEFAULT_BUTTON_PIN)
