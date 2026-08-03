"""Moteur d'inférence du worker de pré-annotation.

Une interface unique et remplaçable : la version de YOLO n'est pas figée, et
les tests utilisent un double — la suite ne dépend jamais des poids réels.
L'implémentation ultralytics n'importe ses dépendances (torch) qu'à
l'instanciation : elles vivent dans le venv du worker (requirements-worker.txt),
jamais dans celui du serveur web.

Le moteur ne touche AUCUN fichier image : le worker lit le stockage et passe
une image déjà décodée — c'est ce qui garde le stockage en lecture seule et le
moteur agnostique du mode de stockage.
"""
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("preannotation")


@dataclass(frozen=True)
class Detection:
    """Boîte YOLO normalisée [0,1], relative au fichier soumis à l'inférence
    (le crop s'il existe, l'original sinon)."""

    class_id: int
    x_center: float
    y_center: float
    box_width: float
    box_height: float
    confidence: float


class InferenceEngine(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifiant des poids réellement chargés (`nom@empreinte-courte`)."""

    @abstractmethod
    def infer(self, image) -> list[Detection]:
        """`image` : PIL.Image déjà décodée."""


def _borne(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def classes_non_proposables(noms_modele, noms_referentiel) -> tuple[str, ...]:
    """Garde de compatibilité poids ↔ data.yaml : règle du PRÉFIXE EXACT.

    Des poids peuvent sortir moins de classes que le référentiel, à condition
    que leurs noms en soient exactement le préfixe (ordre et orthographe) :
    les ids émis gardent alors le même sens. Tout renommage, réordonnancement
    ou classe surnuméraire est refusé. Renvoie les classes que le modèle ne
    pourra jamais proposer — à journaliser NOMMÉMENT au démarrage.

    Accepter un préfixe strict plutôt qu'une égalité est un assouplissement
    TRANSITOIRE (validé le 2026-07-31) : Eponge n'a aucun exemple annoté, des
    poids réentraînés en 8 classes ne la détecteraient pas davantage — exiger
    l'égalité bloquerait la file en attendant un modèle qui n'apporterait
    rien. La garde redevient une équivalence d'elle-même dès qu'un
    réentraînement produit toutes les classes."""
    if len(noms_modele) > len(noms_referentiel):
        raise ValueError(
            f"les poids sortent {len(noms_modele)} classes, le référentiel "
            f"data.yaml n'en compte que {len(noms_referentiel)} — classes "
            "surnuméraires, poids refusés")
    for i, (modele, attendu) in enumerate(zip(noms_modele, noms_referentiel)):
        if modele != attendu:
            raise ValueError(
                f"classe {i} : « {modele} » dans les poids, « {attendu} » "
                "dans data.yaml — renommage ou réordonnancement, poids refusés")
    return tuple(noms_referentiel[len(noms_modele):])


class UltralyticsEngine(InferenceEngine):
    """YOLO via ultralytics.

    Les poids sont identifiés par `nom@sha256-court`. Si le fichier de poids
    change (mtime ou taille) entre deux inférences — best.pt remplacé sans
    redémarrer le worker — modèle ET empreinte sont rechargés : model_name ne
    ment jamais sur les poids qui ont réellement produit les boîtes.
    """

    def __init__(self, weights_path: Path | str, *, conf: float, max_det: int,
                 expected_names: tuple[str, ...]):
        self._weights = Path(weights_path)
        self._conf = conf
        self._max_det = max_det
        self._expected_names = tuple(expected_names)
        self._load()

    def _load(self) -> None:
        from ultralytics import YOLO

        stat = self._weights.stat()
        self._signature = (stat.st_mtime_ns, stat.st_size)
        sha = hashlib.sha256(self._weights.read_bytes()).hexdigest()
        self._model = YOLO(str(self._weights))
        noms_modele = [self._model.names[i]
                       for i in range(len(self._model.names))]
        try:
            self.classes_absentes = classes_non_proposables(
                noms_modele, self._expected_names)
        except ValueError as exc:
            raise ValueError(f"poids {self._weights.name} : {exc}") from None
        self._model_name = f"{self._weights.name}@{sha[:8]}"
        logger.info("poids chargés : %s (%d classes)",
                    self._model_name, len(noms_modele))
        for nom in self.classes_absentes:
            logger.warning(
                "classe « %s » : absente des poids %s — elle ne sera JAMAIS "
                "proposée par la pré-annotation avec ces poids",
                nom, self._model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    def _refresh(self) -> None:
        stat = self._weights.stat()
        if (stat.st_mtime_ns, stat.st_size) != self._signature:
            logger.warning("fichier de poids modifié sur disque — rechargement")
            self._load()

    def infer(self, image) -> list[Detection]:
        self._refresh()
        result = self._model.predict(
            image, conf=self._conf, max_det=self._max_det, verbose=False
        )[0]
        detections = []
        for box in result.boxes:
            x, y, w, h = (float(v) for v in box.xywhn[0])
            detections.append(Detection(
                class_id=int(box.cls),
                x_center=_borne(x, 0.0, 1.0),
                y_center=_borne(y, 0.0, 1.0),
                # la base exige des dimensions strictement positives dans (0,1]
                box_width=_borne(w, 1e-6, 1.0),
                box_height=_borne(h, 1e-6, 1.0),
                confidence=_borne(float(box.conf), 0.0, 1.0),
            ))
        return detections
