"""Worker de pré-annotation : inférence simulée par un double — la suite ne
dépend jamais des poids réels (l'inférence réelle a son test facultatif dans
test_inference_reelle.py)."""
import itertools
import threading
import time

import pytest

from app.inference import Detection, InferenceEngine
from conftest import STORAGE_TEST_ROOT, exec_sql, exec_sql_all, insert_image
from test_import import make_jpg

DETECTION = Detection(class_id=0, x_center=0.5, y_center=0.5,
                      box_width=0.2, box_height=0.2, confidence=0.42)


class FakeEngine(InferenceEngine):
    """Double : boîtes déterministes, pannes programmables par index d'appel
    (1-based ; "*" = toujours), enregistre la taille des images reçues."""

    def __init__(self, detections=(DETECTION,), delai=0.0, panne_sur=frozenset()):
        self.detections = list(detections)
        self.delai = delai
        self.panne_sur = set(panne_sur)
        self.appels = []

    @property
    def model_name(self):
        return "fake.pt@feedface"

    def infer(self, image):
        self.appels.append(image.size)
        if "*" in self.panne_sur or len(self.appels) in self.panne_sur:
            raise RuntimeError("panne simulée")
        if self.delai:
            time.sleep(self.delai)
        return list(self.detections)


def make_worker(moteur, **kw):
    from app.db import get_sessionmaker
    from app.storage import get_storage
    from app.worker import Worker

    params = dict(max_attempts=3, poll_seconds=0.01,
                  max_consecutive_failures=5)
    params.update(kw)
    return Worker(sessionmaker=get_sessionmaker(), storage=get_storage(),
                  engine=moteur, **params)


_compteur_sha = itertools.count(1)


def image_en_attente(engine, base_ids, name="a.jpg", **cols):
    """Ligne images en attente + son fichier réel dans le stockage."""
    make_jpg(STORAGE_TEST_ROOT / "p", name, "red")
    sha = f"{next(_compteur_sha):02x}" * 32
    return insert_image(engine, base_ids, sha, f"p/{name}", **cols)


def test_pre_annotation_nominale(engine, base_ids):
    imgs = [image_en_attente(engine, base_ids, f"n{i}.jpg") for i in range(2)]
    fake = FakeEngine(detections=[DETECTION, DETECTION])
    worker = make_worker(fake)
    worker.run(drain=True)

    assert worker.stats.traitees == 2 and worker.stats.echecs == 0
    for img in imgs:
        assert exec_sql(engine, "SELECT status FROM images WHERE id = %s",
                        (img,)) == "a_annoter"
        boites = exec_sql_all(engine,
                              "SELECT source, state, confidence, model_name"
                              " FROM annotations WHERE image_id = %s", (img,))
        assert len(boites) == 2
        assert all(b == ("modele", "proposee", 0.42, "fake.pt@feedface")
                   for b in boites)
    # transition tracée, auteur NULL : c'est le système
    assert exec_sql(engine,
                    "SELECT count(*) FROM image_status_events"
                    " WHERE from_status = 'en_attente_preannotation'"
                    " AND to_status = 'a_annoter' AND changed_by IS NULL") == 2
    assert exec_sql(engine, "SELECT count(*) FROM images"
                    " WHERE preannotation_error IS NOT NULL") == 0


def test_reprise_ecrase_les_proposees_du_modele(engine, base_ids):
    """Boîtes proposées laissées par un run interrompu ou d'anciens poids :
    écrasées à la reprise — jamais complétées (doublons), jamais gardées
    (qualité inconnue, model_name mélangés)."""
    img = image_en_attente(engine, base_ids)
    for _ in range(2):
        exec_sql(engine,
                 "INSERT INTO annotations (image_id, class_id, x_center,"
                 " y_center, box_width, box_height, source, state, confidence,"
                 " model_name) VALUES (%s, 1, 0.3, 0.3, 0.1, 0.1, 'modele',"
                 " 'proposee', 0.2, 'ancien.pt@deadbeef') RETURNING id", (img,))

    make_worker(FakeEngine()).run(drain=True)

    boites = exec_sql_all(engine,
                          "SELECT model_name FROM annotations"
                          " WHERE image_id = %s", (img,))
    assert [b[0] for b in boites] == ["fake.pt@feedface"]  # ni doublon ni relique


def test_invariant_viole_gare_immediatement(engine, base_ids):
    """Annotation humaine ou décidée sur une image en attente : le worker ne
    devine pas, il gare l'image sans y toucher."""
    img = image_en_attente(engine, base_ids)
    exec_sql(engine,
             "INSERT INTO annotations (image_id, class_id, x_center, y_center,"
             " box_width, box_height, source, state, created_by, decided_by,"
             " decided_at) VALUES (%s, 0, 0.5, 0.5, 0.1, 0.1, 'humain',"
             " 'validee', %s, %s, now()) RETURNING id",
             (img, base_ids["alice"], base_ids["alice"]))

    fake = FakeEngine()
    worker = make_worker(fake)
    worker.run(drain=True)

    assert fake.appels == []  # l'inférence n'a même pas été tentée
    ligne = exec_sql_all(engine,
                         "SELECT status, preannotation_attempts,"
                         " preannotation_error_kind FROM images"
                         " WHERE id = %s", (img,))[0]
    assert ligne == ("en_attente_preannotation", 3, "invariant_viole")
    # la boîte humaine est intacte, aucune boîte modèle ajoutée
    assert exec_sql(engine, "SELECT count(*) FROM annotations"
                    " WHERE image_id = %s", (img,)) == 1
    assert exec_sql(engine, "SELECT count(*) FROM image_status_events"
                    " WHERE image_id = %s", (img,)) == 0


def test_ligne_archivee_jamais_traitee(engine, base_ids):
    img = image_en_attente(engine, base_ids)
    exec_sql(engine, "UPDATE images SET superseded_at = now()"
             " WHERE id = %s", (img,))

    fake = FakeEngine()
    make_worker(fake).run(drain=True)

    assert fake.appels == []
    assert exec_sql(engine, "SELECT status FROM images WHERE id = %s",
                    (img,)) == "en_attente_preannotation"
    assert exec_sql(engine, "SELECT count(*) FROM annotations") == 0


def test_echec_gare_au_plafond_puis_deblocage(engine, base_ids):
    img = image_en_attente(engine, base_ids)
    worker = make_worker(FakeEngine(panne_sur={"*"}),
                         max_consecutive_failures=10)
    worker.run(drain=True)  # 3 tentatives, puis la file est vide : garée

    ligne = exec_sql_all(engine,
                         "SELECT status, preannotation_attempts,"
                         " preannotation_error_kind, preannotation_error"
                         " FROM images WHERE id = %s", (img,))[0]
    assert ligne == ("en_attente_preannotation", 3, "moteur_indisponible",
                     "panne simulée")
    assert worker.stats.echecs == 3
    assert exec_sql(engine, "SELECT count(*) FROM annotations") == 0

    # geste admin : remettre le compteur pour retenter, moteur réparé
    exec_sql(engine, "UPDATE images SET preannotation_attempts = 0"
             " WHERE id = %s", (img,))
    make_worker(FakeEngine()).run(drain=True)
    assert exec_sql(engine, "SELECT status FROM images WHERE id = %s",
                    (img,)) == "a_annoter"
    assert exec_sql(engine, "SELECT preannotation_error FROM images"
                    " WHERE id = %s", (img,)) is None  # le succès efface l'échec


def test_fichier_absent_du_stockage(engine, base_ids):
    img = insert_image(engine, base_ids, "d" * 64, "p/fantome.jpg")
    fake = FakeEngine()
    make_worker(fake).run(drain=True)

    assert fake.appels == []
    assert exec_sql(engine, "SELECT preannotation_error_kind FROM images"
                    " WHERE id = %s", (img,)) == "fichier_illisible"


def test_panne_systemique_arrete_le_worker(engine, base_ids):
    """Des poids cassés font échouer image après image : le worker doit
    s'arrêter au lieu de garer toute la file."""
    from app.worker import PanneWorker

    imgs = [image_en_attente(engine, base_ids, f"s{i}.jpg") for i in range(3)]
    worker = make_worker(FakeEngine(panne_sur={"*"}),
                         max_consecutive_failures=5)
    with pytest.raises(PanneWorker):
        worker.run(drain=True)

    # img1 : 3 tentatives (garée), img2 : 2 tentatives → 5 échecs consécutifs,
    # arrêt AVANT de toucher img3
    tentatives = [exec_sql(engine, "SELECT preannotation_attempts FROM images"
                           " WHERE id = %s", (i,)) for i in imgs]
    assert tentatives == [3, 2, 0]


def test_transaction_par_image(engine, base_ids):
    """Un incident sur une image ne perd pas le travail déjà commité."""
    ok, casse = (image_en_attente(engine, base_ids, n)
                 for n in ("ok.jpg", "casse.jpg"))
    # 1er appel (ok.jpg) passe, tous les suivants plantent
    worker = make_worker(FakeEngine(panne_sur={2, 3, 4}),
                         max_consecutive_failures=10)
    worker.run(drain=True)

    assert exec_sql(engine, "SELECT status FROM images WHERE id = %s",
                    (ok,)) == "a_annoter"
    assert exec_sql(engine, "SELECT count(*) FROM annotations"
                    " WHERE image_id = %s", (ok,)) == 1
    assert exec_sql_all(engine,
                        "SELECT status, preannotation_attempts FROM images"
                        " WHERE id = %s", (casse,))[0] == \
        ("en_attente_preannotation", 3)


def test_deux_workers_en_parallele(engine, base_ids):
    """Deux workers simultanés : SKIP LOCKED garantit qu'aucune image n'est
    traitée deux fois — aucune boîte en double, un seul événement par image."""
    for i in range(6):
        image_en_attente(engine, base_ids, f"c{i}.jpg")
    fake = FakeEngine(delai=0.05)  # force le chevauchement des deux workers
    w1, w2 = make_worker(fake), make_worker(fake)
    threads = [threading.Thread(target=w.run, kwargs={"drain": True})
               for w in (w1, w2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert w1.stats.traitees + w2.stats.traitees == 6
    assert exec_sql(engine, "SELECT count(*) FROM images"
                    " WHERE status = 'a_annoter'") == 6
    # exactement UNE boîte et UN événement par image — jamais deux jeux
    assert exec_sql_all(engine,
                        "SELECT image_id, count(*) FROM annotations"
                        " GROUP BY image_id HAVING count(*) <> 1") == []
    assert exec_sql(engine, "SELECT count(*) FROM image_status_events") == 6


def test_le_crop_prime_sur_l_original(engine, base_ids):
    """Les coordonnées sont relatives au fichier annoté : le crop s'il
    existe. Le double vérifie quelle image il a reçue par sa taille."""
    from PIL import Image as PILImage

    (STORAGE_TEST_ROOT / "c").mkdir(exist_ok=True)
    PILImage.new("RGB", (40, 40), "blue").save(STORAGE_TEST_ROOT / "c" / "x.jpg")
    image_en_attente(engine, base_ids, "entier.jpg",
                     cropped_path="c/x.jpg", crop_x=0, crop_y=0,
                     cropped_width=40, cropped_height=40)

    fake = FakeEngine()
    make_worker(fake).run(drain=True)
    assert fake.appels == [(40, 40)]  # le crop (40×40), pas l'original (64×48)


def test_essai_mesure_sans_rien_ecrire(db, engine, base_ids):
    from app.storage import get_storage
    from app.worker import essai

    for i in range(3):
        image_en_attente(engine, base_ids, f"e{i}.jpg")
    confs = [Detection(0, 0.5, 0.5, 0.1, 0.1, c) for c in (0.9, 0.3, 0.07)]
    rapport = essai(db, get_storage(), FakeEngine(detections=confs),
                    echantillon=10, seuils=[0.5, 0.25, 0.05])

    assert "3 image(s)" in rapport
    assert "vérité terrain inconnue" in rapport  # file d'attente : repli
    assert "seuil 0.05 : médiane 3" in rapport
    assert "seuil 0.25 : médiane 2" in rapport
    assert "seuil 0.50 : médiane 1" in rapport
    # lecture seule : aucune boîte, aucun statut, aucun compteur modifié
    assert exec_sql(engine, "SELECT count(*) FROM annotations") == 0
    assert exec_sql(engine, "SELECT count(*) FROM images"
                    " WHERE status <> 'en_attente_preannotation'"
                    " OR preannotation_attempts <> 0") == 0


def test_essai_stratifie_porteuses_negatifs(db, engine, base_ids):
    """Sur une base annotée : échantillon tiré à parts égales entre les
    sessions (jamais les N premières images d'un seul poste), et distribution
    séparée porteuses/négatifs — sur un négatif, toute détection est fausse,
    c'est le pire cas pour l'annotateur."""
    from app.storage import get_storage
    from app.worker import essai

    for i in range(3):
        iid1 = image_en_attente(engine, base_ids, f"a{i}.jpg",
                                status="annotee")
        iid2 = image_en_attente(engine, base_ids, f"b{i}.jpg",
                                status="annotee", session_id=base_ids["s2"],
                                batch_id=base_ids["b2"])
        if i == 0:  # une image porteuse par session, les autres = négatifs
            for iid in (iid1, iid2):
                exec_sql(engine,
                         "INSERT INTO annotations (image_id, class_id,"
                         " x_center, y_center, box_width, box_height, source,"
                         " state, created_by, decided_by, decided_at)"
                         " VALUES (%s, 0, 0.5, 0.5, 0.1, 0.1, 'humain',"
                         " 'validee', %s, %s, now()) RETURNING id",
                         (iid, base_ids["alice"], base_ids["alice"]))
    image_en_attente(engine, base_ids, "file.jpg")  # en attente : ignorée

    confs = [Detection(0, 0.5, 0.5, 0.1, 0.1, c) for c in (0.9, 0.3, 0.07)]
    rapport = essai(db, get_storage(), FakeEngine(detections=confs),
                    echantillon=10, seuils=[0.95, 0.05])

    # les annotées priment sur la file, à parts égales entre sessions
    assert "6 image(s) (s1 : 3, s2 : 3)" in rapport
    assert "vérité terrain : 2 porteuse(s) de boîtes, 4 négatif(s)" in rapport
    assert "porteuses (2) : médiane 3" in rapport
    assert "négatifs  (4) : médiane 3" in rapport and "0/4 sans aucune" in rapport
    # au-dessus de toute confiance : plus une seule fausse boîte
    assert "4/4 sans aucune" in rapport

    # échantillon plus petit que la base : l'équilibre est préservé
    rapport4 = essai(db, get_storage(), FakeEngine(detections=confs),
                     echantillon=4, seuils=[0.05])
    assert "4 image(s) (s1 : 2, s2 : 2)" in rapport4

    # lecture seule : rien d'autre que les 2 boîtes du décor
    assert exec_sql(engine, "SELECT count(*) FROM annotations") == 2
