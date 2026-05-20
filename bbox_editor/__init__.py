"""
Composant Streamlit d'annotation de bounding boxes (Konva/React).

Fonctionnalités :
  - Mode Transform : dessiner de nouvelles bbox + sélectionner / déplacer / redimensionner
  - Mode Del       : clic sur une bbox pour la supprimer
  - Sélecteur de classe intégré + bouton Complete pour valider
"""
import os
from hashlib import md5

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from packaging import version
from PIL import Image
from streamlit.components.v1.components import CustomComponent

try:
    from streamlit.elements.image import image_to_url
except ImportError:
    from streamlit.elements.lib.image_utils import image_to_url

_STREAMLIT_VERSION = version.parse(st.__version__)
_USE_LAYOUT_CONFIG = _STREAMLIT_VERSION >= version.parse("1.49.0")
if _USE_LAYOUT_CONFIG:
    from streamlit.elements.lib.layout_utils import LayoutConfig

_BUILD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "build")
_component_func = components.declare_component("st-bbox-editor", path=_BUILD_PATH)


def detection(
    image,
    label_list: list,
    bboxes: list,
    labels: list,
    height: int = 512,
    width: int = 512,
    line_width: float = 2.0,
    color_map: dict = None,
    use_space: bool = False,
    key: str = None,
    token: int = 0,
) -> list | None:
    """
    Affiche un éditeur de bounding boxes interactif.

    Paramètres
    ----------
    image       : PIL.Image ou np.ndarray (RGB)
    label_list  : liste ordonnée des noms de classes
    bboxes      : list de [x, y, w, h] en pixels (coin haut-gauche)
    labels      : list d'indices de classe correspondants
    height/width: dimensions max d'affichage (le composant adapte le scale)
    line_width  : épaisseur des traits
    color_map   : dict {label: '#hexcolor'} ; généré automatiquement si None
    use_space   : valider avec Espace en plus du bouton Complete
    key         : clé Streamlit unique

    Retour
    ------
    None tant que l'utilisateur n'a pas cliqué "Complete".
    List de {'bbox': [x, y, w, h], 'label_id': int, 'label': str} sinon.
    """
    if isinstance(image, np.ndarray):
        pil = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        pil = image.copy()
    else:
        pil = Image.open(image)

    original_size = pil.size
    pil.thumbnail((width, height))
    scale = original_size[0] / pil.size[0]

    if _USE_LAYOUT_CONFIG:
        layout_cfg = LayoutConfig(width=pil.size[0], height=pil.size[1])
        image_url = image_to_url(
            pil, layout_cfg, True, "RGB", "PNG",
            f"bbox-{md5(pil.tobytes()).hexdigest()}-{key}",
        )
    else:
        image_url = image_to_url(
            pil, pil.size[0], True, "RGB", "PNG",
            f"bbox-{md5(pil.tobytes()).hexdigest()}-{key}",
        )

    if color_map is None:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap("gist_rainbow")
        color_map = {
            l: "#%02x%02x%02x" % tuple(
                int(d) for d in np.array(cmap(i / len(label_list))) * 255
            )[:3]
            for i, l in enumerate(label_list)
        }

    bbox_info = [
        {
            "bbox": [b / scale for b in bbox],
            "label_id": label_idx,
            "label": label_list[label_idx],
        }
        for bbox, label_idx in zip(bboxes, labels)
    ]

    raw = _component_func(
        image_url=image_url,
        image_size=list(pil.size),
        label_list=label_list,
        bbox_info=bbox_info,
        color_map=color_map,
        line_width=line_width,
        use_space=use_space,
        token=token,
        key=key,
    )

    if raw is None:
        return None
    if not isinstance(raw, dict) or raw.get("token") != token:
        return None

    return [
        {
            "bbox": [b * scale for b in item["bbox"]],
            "label_id": item["label_id"],
            "label": item["label"],
        }
        for item in raw["bboxes"]
    ]
