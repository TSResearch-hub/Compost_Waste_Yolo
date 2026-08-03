"""La liste blanche des transitions vit en deux endroits : le trigger CW002
(migration 0001) et models.IMAGE_STATUS_TRANSITIONS. Ce test compare le
comportement RÉEL du trigger, paire par paire, à la liste Python : toute
divergence entre les deux exemplaires le fait échouer."""
from itertools import product

from sqlalchemy.exc import DBAPIError

from conftest import insert_image
from app.models import IMAGE_STATUS_TRANSITIONS, IMAGE_STATUSES


def test_trigger_identique_a_la_liste_python(engine, base_ids):
    autorisees_par_le_trigger = set()

    for i, (frm, to) in enumerate(product(IMAGE_STATUSES, repeat=2)):
        if frm == to:
            continue
        sha = f"{i:064x}"
        img = insert_image(engine, base_ids, sha, f"p/{i}.jpg")

        # Forcer l'état de départ, trigger désactivé (les CHECK restent actifs :
        # 'relue' exige un relecteur)
        with engine.begin() as c:
            c.exec_driver_sql(
                "ALTER TABLE images DISABLE TRIGGER images_transition_statut")
            if frm == "relue":
                c.exec_driver_sql(
                    "UPDATE images SET status = %s, reviewed_by = %s,"
                    " reviewed_at = now() WHERE id = %s",
                    (frm, base_ids["bob"], img))
            else:
                c.exec_driver_sql(
                    "UPDATE images SET status = %s WHERE id = %s", (frm, img))
            c.exec_driver_sql(
                "ALTER TABLE images ENABLE TRIGGER images_transition_statut")

        # Tenter la transition, trigger actif
        try:
            with engine.begin() as c:
                if to == "relue":
                    c.exec_driver_sql(
                        "UPDATE images SET status = %s, reviewed_by = %s,"
                        " reviewed_at = now() WHERE id = %s",
                        (to, base_ids["bob"], img))
                else:
                    c.exec_driver_sql(
                        "UPDATE images SET status = %s WHERE id = %s", (to, img))
            autorisees_par_le_trigger.add((frm, to))
        except DBAPIError as e:
            assert e.orig.sqlstate == "CW002", (
                f"{frm} -> {to} : refus pour une autre raison que le trigger "
                f"({e.orig.sqlstate})")

    assert autorisees_par_le_trigger == set(IMAGE_STATUS_TRANSITIONS), (
        "le trigger CW002 et models.IMAGE_STATUS_TRANSITIONS ont divergé — "
        "mettre à jour LES DEUX (migration et modèles)")
