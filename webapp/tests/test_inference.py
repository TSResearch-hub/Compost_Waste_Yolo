"""Garde de compatibilité poids ↔ data.yaml : règle du préfixe exact
(assouplissement transitoire validé le 2026-07-31 — Eponge sans exemple).
Fonction pure : testable sans ultralytics ni poids réels."""
import pytest

from app.inference import classes_non_proposables

REF = ("Plastique", "Metal", "Carton", "Aluminium", "Ceramique", "Verre",
       "Composite", "Eponge")


def test_correspondance_exacte():
    assert classes_non_proposables(list(REF), REF) == ()


def test_prefixe_exact_nomme_les_absentes():
    assert classes_non_proposables(list(REF[:7]), REF) == ("Eponge",)
    assert classes_non_proposables(list(REF[:5]), REF) == (
        "Verre", "Composite", "Eponge")


def test_renommage_refuse():
    noms = list(REF[:7])
    noms[1] = "Métal"  # l'accent suffit : l'orthographe fait foi
    with pytest.raises(ValueError, match="renommage ou réordonnancement"):
        classes_non_proposables(noms, REF)


def test_reordonnancement_refuse():
    noms = list(REF[:7])
    noms[0], noms[1] = noms[1], noms[0]
    with pytest.raises(ValueError, match="renommage ou réordonnancement"):
        classes_non_proposables(noms, REF)


def test_classe_surnumeraire_refusee():
    with pytest.raises(ValueError, match="surnuméraires"):
        classes_non_proposables(list(REF) + ["Papier"], REF)
