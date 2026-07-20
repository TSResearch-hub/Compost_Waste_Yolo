"""Tests du référentiel de classes et de la classification d'alertes.

Contexte : le modèle actuel (weights/best.pt) nomme ses classes SANS accents
(Metal, Ceramique, Eponge) ; le référentiel du projet (CLASS_MAP) reprend ses
8 classes dans le même ordre, mais accentuées. Ces tests verrouillent la
tolérance aux accents qui a corrigé deux bugs réels (pré-annotations classées
Aluminium, alertes du direct inversées).
"""
from types import SimpleNamespace

import numpy as np

import helper

# Noms de classes tels que les renvoie réellement weights/best.pt
MODEL_NAMES = {0: "Plastique", 1: "Metal", 2: "Carton", 3: "Aluminium",
               4: "Ceramique", 5: "Verre", 6: "Composite", 7: "Eponge"}


class _FakeTensor:
    """Imite tensor.cpu().numpy() d'ultralytics."""
    def __init__(self, arr):
        self._arr = np.asarray(arr, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


def _fake_results(xywh, cls_ids, names=MODEL_NAMES, conf=None):
    if conf is None:
        conf = [1.0] * len(cls_ids)
    return SimpleNamespace(
        boxes=SimpleNamespace(xywh=_FakeTensor(xywh), cls=_FakeTensor(cls_ids),
                              conf=_FakeTensor(conf)),
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
    assert helper.class_name_to_id("Eponge") == helper.CLASS_MAP["Éponge"]


def test_mapping_toutes_les_classes_du_modele_actuel():
    attendu = {"Plastique": 0, "Metal": 1, "Carton": 2, "Aluminium": 3,
               "Ceramique": 4, "Verre": 5, "Composite": 6, "Eponge": 7}
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


def test_seuils_par_classe_filtrent_la_pre_annotation():
    # Metal 0.9 (gardé), Ceramique 0.3 (gardée : seuil 0.25), Metal 0.2 (écarté : seuil 0.5)
    res = _fake_results(
        xywh=[[100, 100, 40, 20], [300, 200, 60, 80], [50, 50, 10, 10]],
        cls_ids=[1, 4, 1],
        conf=[0.9, 0.3, 0.2],
    )
    conf_by_id = {helper.CLASS_MAP["Métal"]: 0.5, helper.CLASS_MAP["Céramique"]: 0.25}
    bboxes, labels = helper.get_detection_initial_data(res, conf_by_id)
    assert labels == [helper.CLASS_MAP["Métal"], helper.CLASS_MAP["Céramique"]]
    assert len(bboxes) == 2


def test_sans_seuils_par_classe_tout_est_garde():
    res = _fake_results(xywh=[[10, 10, 4, 4]], cls_ids=[1], conf=[0.05])
    _, labels = helper.get_detection_initial_data(res)
    assert labels == [helper.CLASS_MAP["Métal"]]


# ── classify_waste_type ──────────────────────────────────────────────────────

def test_plastique_est_non_compostable():
    # Bug historique : plastique s'affichait « COMPOSTABLE » en vert
    compost, non_compost, risk, danger = helper.classify_waste_type({"Plastique"})
    assert compost == set()
    assert non_compost == {"Plastique"}


def test_classe_hors_referentiel_sans_alerte():
    # Le modèle 8 classes ne détecte que des intrus : un nom hors des listes
    # d'alerte ne doit rien déclencher (COMPOSTABLE est vide aujourd'hui).
    assert helper.classify_waste_type({"Inconnu"}) == (set(), set(), set(), set())


def test_alertes_tolerantes_aux_accents_du_modele():
    # Noms SANS accents comme les renvoie le modèle : ils ne matchaient plus
    # aucune liste avant le correctif (aucune alerte affichée)
    compost, non_compost, risk, danger = helper.classify_waste_type(
        {"Metal", "Ceramique", "Verre", "Plastique", "Eponge"}
    )
    assert risk == {"Metal"}
    assert danger == {"Ceramique", "Verre"}
    assert non_compost == {"Plastique", "Eponge"}
    assert compost == set()
