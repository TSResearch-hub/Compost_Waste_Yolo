"""Tests du référentiel de classes et de la classification d'alertes.

Contexte : le modèle actuel (weights/best.pt) nomme ses classes SANS accents
(Metal, Ceramique) et n'en a que 7, alors que le référentiel du projet
(CLASS_MAP) en a 9 avec accents. Ces tests verrouillent la tolérance aux
accents qui a corrigé deux bugs réels (pré-annotations classées Aluminium,
alertes du direct inversées).
"""
from types import SimpleNamespace

import numpy as np

import helper

# Noms de classes tels que les renvoie réellement weights/best.pt
MODEL_NAMES = {0: "Plastique", 1: "Metal", 2: "Carton", 3: "Aluminium",
               4: "Ceramique", 5: "Verre", 6: "Composite"}


class _FakeTensor:
    """Imite tensor.cpu().numpy() d'ultralytics."""
    def __init__(self, arr):
        self._arr = np.asarray(arr, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


def _fake_results(xywh, cls_ids, names=MODEL_NAMES):
    return SimpleNamespace(
        boxes=SimpleNamespace(xywh=_FakeTensor(xywh), cls=_FakeTensor(cls_ids)),
        names=names,
    )


# ── normalize_class_name / class_name_to_id ──────────────────────────────────

def test_normalisation_accents_et_casse():
    assert helper.normalize_class_name("Métal") == "metal"
    assert helper.normalize_class_name("Ceramique") == "ceramique"
    assert helper.normalize_class_name("  VERRE ") == "verre"


def test_mapping_noms_modele_sans_accents():
    # Les 4 classes qui tombaient toutes en Aluminium (3) avant le correctif
    assert helper.class_name_to_id("Metal") == helper.CLASS_MAP["Métal"]
    assert helper.class_name_to_id("Ceramique") == helper.CLASS_MAP["Céramique"]
    assert helper.class_name_to_id("Verre") == helper.CLASS_MAP["Verre"]
    assert helper.class_name_to_id("Composite") == helper.CLASS_MAP["Composite"]


def test_mapping_toutes_les_classes_du_modele_actuel():
    attendu = {"Plastique": 0, "Metal": 1, "Carton": 2, "Aluminium": 3,
               "Ceramique": 4, "Verre": 7, "Composite": 8}
    for nom, cid in attendu.items():
        assert helper.class_name_to_id(nom) == cid, nom


def test_mapping_nom_inconnu():
    assert helper.class_name_to_id("Inconnu") is None


# ── get_detection_initial_data ───────────────────────────────────────────────

def test_pre_annotations_classees_correctement():
    # Une bbox Metal (id modèle 1) et une Ceramique (id modèle 4)
    res = _fake_results(xywh=[[100, 100, 40, 20], [300, 200, 60, 80]], cls_ids=[1, 4])
    bboxes, labels = helper.get_detection_initial_data(res)
    assert labels == [helper.CLASS_MAP["Métal"], helper.CLASS_MAP["Céramique"]]
    # Conversion centre → coin haut-gauche
    assert bboxes[0] == [80.0, 90.0, 40.0, 20.0]


def test_pre_annotation_classe_inconnue_fallback_premiere_classe():
    res = _fake_results(xywh=[[10, 10, 4, 4]], cls_ids=[0], names={0: "Extraterrestre"})
    _, labels = helper.get_detection_initial_data(res)
    assert labels == [0]


def test_resultats_vides():
    assert helper.get_detection_initial_data(None) == ([], [])


# ── classify_waste_type ──────────────────────────────────────────────────────

def test_plastique_est_non_compostable():
    # Bug historique : plastique s'affichait « COMPOSTABLE » en vert
    compost, non_compost, risk, danger = helper.classify_waste_type({"Plastique"})
    assert compost == set()
    assert non_compost == {"Plastique"}


def test_organique_est_compostable():
    compost, non_compost, _, _ = helper.classify_waste_type({"Organique"})
    assert compost == {"Organique"}
    assert non_compost == set()


def test_alertes_tolerantes_aux_accents_du_modele():
    # Noms SANS accents comme les renvoie le modèle : ils ne matchaient plus
    # aucune liste avant le correctif (aucune alerte affichée)
    compost, non_compost, risk, danger = helper.classify_waste_type(
        {"Metal", "Ceramique", "Verre", "Plastique"}
    )
    assert risk == {"Metal"}
    assert danger == {"Ceramique", "Verre"}
    assert non_compost == {"Plastique"}
    assert compost == set()
