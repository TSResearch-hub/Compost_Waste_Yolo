"""Garanties portées par la base elle-même (conversion des 39 assertions de
validation du segment 1). Chaque interdit est vérifié par son SQLSTATE exact :
23505 unicité, 23514 CHECK, 23503 FK, CW001/CW002 triggers."""
from conftest import exec_sql, insert_image, refus

SHA_A = "a" * 64
SHA_B = "b" * 64


def test_zero_derive_modeles_base(engine):
    """La base migrée correspond exactement aux modèles (équivalent
    `alembic check` — attrape toute dérive modèles ↔ migration 0001)."""
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    from app.db import Base

    with engine.connect() as conn:
        ctx = MigrationContext.configure(conn)
        diff = compare_metadata(ctx, Base.metadata)
    assert diff == [], f"dérive modèles/base : {diff}"


def test_referentiel_utilisateurs(engine, base_ids):
    refus(engine, "23505",
          "INSERT INTO users (username, password_hash, role)"
          " VALUES ('ADMIN', 'x', 'annotateur')")  # unicité insensible à la casse
    refus(engine, "23514",
          "INSERT INTO users (username, password_hash, role)"
          " VALUES ('eve', 'x', 'superuser')")  # rôle hors liste


def test_lots_et_fk_composite(engine, base_ids):
    # même nom de lot autorisé dans deux sessions, refusé dans la même
    exec_sql(engine,
             "INSERT INTO batches (session_id, name, created_by)"
             " VALUES (%s, 'lot 2', %s) RETURNING id",
             (base_ids["s1"], base_ids["admin"]))
    refus(engine, "23505",
          "INSERT INTO batches (session_id, name, created_by)"
          " VALUES (%s, 'lot 1', %s)", (base_ids["s1"], base_ids["admin"]))
    # une image ne peut pas rattacher une session à un lot d'une autre session
    insert_image(engine, base_ids, SHA_A, "p/a.jpg")
    refus(engine, "23503",
          "INSERT INTO images (session_id, batch_id, original_filename,"
          " export_filename, source_label, original_path, width, height, sha256)"
          " VALUES (%s, %s, 'b.jpg', 'b.jpg', 'p', 'p/b.jpg', 100, 100, %s)",
          (base_ids["s1"], base_ids["b2"], SHA_B))


def test_machine_a_etats_parcours_nominal(engine, base_ids):
    img = insert_image(engine, base_ids, SHA_A, "p/a.jpg")
    refus(engine, "CW002", "UPDATE images SET status = 'relue' WHERE id = %s", (img,))
    refus(engine, "CW002", "UPDATE images SET status = 'annotee' WHERE id = %s", (img,))
    for to in ("a_annoter", "en_cours"):
        exec_sql(engine, "UPDATE images SET status = %s WHERE id = %s", (to, img))
    exec_sql(engine,
             "UPDATE images SET status = 'annotee', annotated_by = %s,"
             " annotated_at = now() WHERE id = %s", (base_ids["alice"], img))
    refus(engine, "23514",  # relue sans relecteur
          "UPDATE images SET status = 'relue' WHERE id = %s", (img,))
    refus(engine, "23514",  # relecteur = annotateur
          "UPDATE images SET status = 'relue', reviewed_by = %s,"
          " reviewed_at = now() WHERE id = %s", (base_ids["alice"], img))
    exec_sql(engine,
             "UPDATE images SET status = 'relue', reviewed_by = %s,"
             " reviewed_at = now() WHERE id = %s", (base_ids["bob"], img))
    # réouverture : la relecture est annulée
    exec_sql(engine,
             "UPDATE images SET status = 'en_cours', reviewed_by = NULL,"
             " reviewed_at = NULL WHERE id = %s", (img,))


def test_annotations_coherences(engine, base_ids):
    img = insert_image(engine, base_ids, SHA_A, "p/a.jpg")
    ann = exec_sql(engine,
                   "INSERT INTO annotations (image_id, class_id, x_center, y_center,"
                   " box_width, box_height, source, confidence, model_name)"
                   " VALUES (%s, 3, 0.5, 0.5, 0.2, 0.2, 'modele', 0.87, 'best.pt')"
                   " RETURNING id", (img,))
    refus(engine, "23514",  # boîte humaine sans auteur
          "INSERT INTO annotations (image_id, class_id, x_center, y_center,"
          " box_width, box_height, source)"
          " VALUES (%s, 0, 0.5, 0.5, 0.1, 0.1, 'humain')", (img,))
    refus(engine, "23514",  # boîte modèle avec created_by
          "INSERT INTO annotations (image_id, class_id, x_center, y_center,"
          " box_width, box_height, source, created_by)"
          " VALUES (%s, 0, 0.5, 0.5, 0.1, 0.1, 'modele', %s)",
          (img, base_ids["alice"]))
    refus(engine, "23514",  # validee sans décideur
          "UPDATE annotations SET state = 'validee' WHERE id = %s", (ann,))
    exec_sql(engine,
             "UPDATE annotations SET state = 'validee', decided_by = %s,"
             " decided_at = now() WHERE id = %s", (base_ids["alice"], ann))
    refus(engine, "23514",  # coordonnées hors [0,1]
          "INSERT INTO annotations (image_id, class_id, x_center, y_center,"
          " box_width, box_height, source, created_by)"
          " VALUES (%s, 0, 1.2, 0.5, 0.1, 0.1, 'humain', %s)",
          (img, base_ids["alice"]))


def test_gel_du_crop(engine, base_ids):
    img = insert_image(engine, base_ids, SHA_A, "p/a.jpg")
    exec_sql(engine,
             "INSERT INTO annotations (image_id, class_id, x_center, y_center,"
             " box_width, box_height, source, confidence)"
             " VALUES (%s, 0, 0.5, 0.5, 0.2, 0.2, 'modele', 0.5) RETURNING id",
             (img,))
    refus(engine, "CW001",  # gel dès la première annotation, même 'proposee'
          "UPDATE images SET cropped_path = 'c/a.jpg', crop_x = 0, crop_y = 0,"
          " cropped_width = 50, cropped_height = 50 WHERE id = %s", (img,))
    img2 = insert_image(engine, base_ids, SHA_B, "p/b.jpg")
    exec_sql(engine,  # modifiable tant qu'aucune annotation
             "UPDATE images SET cropped_path = 'c/b.jpg', crop_x = 0, crop_y = 0,"
             " cropped_width = 50, cropped_height = 50 WHERE id = %s", (img2,))
    refus(engine, "23514",  # champs de crop indissociables
          "UPDATE images SET crop_x = NULL WHERE id = %s", (img2,))


def test_recadrage_et_unicite_partielle(engine, base_ids):
    img = insert_image(engine, base_ids, SHA_A, "p/a.jpg")
    refus(engine, "23505",  # réimport du même fichier
          "INSERT INTO images (session_id, batch_id, original_filename,"
          " export_filename, source_label, original_path, width, height, sha256)"
          " VALUES (%s, %s, 'a2.jpg', 'a2.jpg', 'p', 'p/a2.jpg', 100, 100, %s)",
          (base_ids["s1"], base_ids["b1"], SHA_A))
    refus(engine, "23514",  # pointeur sans archivage
          "UPDATE images SET superseded_by_id = %s WHERE id = %s", (img, img))
    # séquence de recadrage : archivage → nouvelle ligne → pointeur, une transaction
    with engine.begin() as c:
        c.exec_driver_sql(
            "UPDATE images SET superseded_at = now() WHERE id = %s", (img,))
        new_id = c.exec_driver_sql(
            "INSERT INTO images (session_id, batch_id, original_filename,"
            " export_filename, source_label, original_path, width, height,"
            " sha256, cropped_path, crop_x, crop_y, cropped_width, cropped_height)"
            " VALUES (%s, %s, 'a.jpg', 'a.jpg', 'p', 'p/a.jpg', 100, 100, %s,"
            " 'c/a_v2.jpg', 10, 10, 80, 80) RETURNING id",
            (base_ids["s1"], base_ids["b1"], SHA_A)).scalar()
        c.exec_driver_sql(
            "UPDATE images SET superseded_by_id = %s WHERE id = %s", (new_id, img))
    # les lignes archivées sont hors des files de travail
    actives = exec_sql(engine,
                       "SELECT count(*) FROM images WHERE superseded_at IS NULL")
    assert actives == 1


def test_verrou_de_lot(engine, base_ids):
    exec_sql(engine,
             "INSERT INTO batch_assignments (batch_id, user_id, assigned_by)"
             " VALUES (%s, %s, %s) RETURNING id",
             (base_ids["b1"], base_ids["alice"], base_ids["admin"]))
    refus(engine, "23505",  # un seul annotateur actif par lot
          "INSERT INTO batch_assignments (batch_id, user_id, assigned_by)"
          " VALUES (%s, %s, %s)",
          (base_ids["b1"], base_ids["bob"], base_ids["admin"]))
    refus(engine, "23514",  # libération sans motif
          "UPDATE batch_assignments SET released_at = now(), released_by = %s"
          " WHERE batch_id = %s AND released_at IS NULL",
          (base_ids["alice"], base_ids["b1"]))
    exec_sql(engine,
             "UPDATE batch_assignments SET released_at = now(), released_by = %s,"
             " release_reason = 'rendu'"
             " WHERE batch_id = %s AND released_at IS NULL RETURNING id",
             (base_ids["alice"], base_ids["b1"]))
    exec_sql(engine,  # réassignation possible après libération
             "INSERT INTO batch_assignments (batch_id, user_id, assigned_by)"
             " VALUES (%s, %s, %s) RETURNING id",
             (base_ids["b1"], base_ids["bob"], base_ids["admin"]))
