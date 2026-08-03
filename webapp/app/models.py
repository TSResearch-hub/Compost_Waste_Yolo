"""Modèles SQLAlchemy du socle — schéma validé le 2026-07-28.

Règles structurantes :
- Les classes d'annotation ne vivent QUE dans data.yaml (DATA_YAML_PATH) :
  seul l'index ``class_id`` est stocké ici ; la borne haute est vérifiée à
  l'applicatif contre data.yaml, jamais recopiée en dur (pas même en CHECK).
- L'id de session reste attaché à l'image jusqu'à l'export ; le split
  train/val/test se fait par session, jamais par image.
- Une pré-annotation (``state='proposee'``) n'est jamais exportable : l'export
  ne prend que ``state='validee'`` sur des images ``annotee`` ou ``relue``.
- Deux triggers PostgreSQL (migration 0001) complètent ces modèles : gel de la
  géométrie de crop dès qu'une image porte une annotation (SQLSTATE CW001) et
  liste blanche des transitions de statut d'image (SQLSTATE CW002).
- Un recadrage ne modifie jamais une ligne ``images`` existante : il en crée
  une nouvelle et archive l'ancienne. L'archivage est porté par
  ``superseded_at`` (c'est lui que testent les index d'unicité partiels) ;
  ``superseded_by_id`` pointe vers la remplaçante et est posé juste après son
  insertion, dans la même transaction — l'ordre inverse est impossible car la
  FK exigerait une ligne qui ne peut pas encore exister tant que l'ancienne
  est active. L'unicité de ``sha256`` et ``original_path`` ne porte que sur
  les lignes actives ; la détection de doublons à l'import, elle, interroge
  TOUTES les lignes.
"""
from datetime import date, datetime

from sqlalchemy import (
    CHAR,
    REAL,
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Double,
    ForeignKey,
    ForeignKeyConstraint,
    Identity,
    Index,
    Integer,
    SmallInteger,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base

ROLES = ("annotateur", "annotateur_confirme", "administrateur")
IMAGE_STATUSES = (
    "en_attente_preannotation",
    "a_annoter",
    "en_cours",
    "annotee",
    "relue",
)
ANNOTATION_SOURCES = ("modele", "humain")
ANNOTATION_STATES = ("proposee", "validee", "rejetee")
RELEASE_REASONS = ("termine", "rendu", "force_admin")
PREANNOTATION_ERROR_KINDS = (
    "fichier_illisible",
    "moteur_indisponible",
    "invariant_viole",
)

# Transitions de statut autorisées — la même liste est gravée dans le trigger
# de la migration 0001 ; toute évolution doit toucher les deux.
IMAGE_STATUS_TRANSITIONS = (
    ("en_attente_preannotation", "a_annoter"),
    ("a_annoter", "en_cours"),
    ("en_cours", "a_annoter"),
    ("en_cours", "annotee"),
    ("annotee", "relue"),
    ("annotee", "en_cours"),
    ("relue", "en_cours"),
)


def _in(values: tuple[str, ...]) -> str:
    return "(" + ", ".join(f"'{v}'" for v in values) + ")"


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint(f"role IN {_in(ROLES)}", name="role_valide"),
        Index("uq_users_username_lower", text("lower(username)"), unique=True),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    username: Mapped[str] = mapped_column(Text)
    password_hash: Mapped[str] = mapped_column(Text)
    display_name: Mapped[str | None] = mapped_column(Text)
    role: Mapped[str] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, server_default=text("true"))
    # NULL uniquement pour le premier administrateur, créé par la CLI
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class AuthSession(Base):
    """Session web (cookie opaque). Seule l'empreinte SHA-256 du jeton est
    stockée : une fuite de la base ne donne aucune session utilisable.
    Expiration glissante : last_seen_at + AUTH_INACTIVITY_MINUTES."""

    __tablename__ = "auth_sessions"
    __table_args__ = (
        Index("ix_auth_sessions_user", "user_id"),
        Index("ix_auth_sessions_last_seen", "last_seen_at"),
    )

    token_hash: Mapped[str] = mapped_column(Text, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class CaptureSession(Base):
    """Session de capture (jour/conditions de prise de vue) — PAS une session
    web. C'est l'unité du split train/val/test."""

    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    name: Mapped[str] = mapped_column(Text, unique=True)
    captured_on: Mapped[date] = mapped_column(Date)
    lighting: Mapped[str | None] = mapped_column(Text)
    camera_height_cm: Mapped[int | None] = mapped_column(Integer)
    compost_state: Mapped[str | None] = mapped_column(Text)
    # Texte libre : l'opérateur terrain n'a pas forcément de compte
    operator: Mapped[str | None] = mapped_column(Text)
    notes: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[int] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Batch(Base):
    __tablename__ = "batches"
    __table_args__ = (
        UniqueConstraint("session_id", "name", name="uq_batches_session_name"),
        # Support de la FK composite depuis images : garantit que l'image et
        # son lot appartiennent à la même session
        UniqueConstraint("id", "session_id", name="uq_batches_id_session"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"))
    name: Mapped[str] = mapped_column(Text)
    created_by: Mapped[int] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class BatchAssignment(Base):
    """Verrou de lot avec historique. Le verrou actif est la ligne non libérée
    (released_at IS NULL) ; l'index unique partiel garantit au plus un
    annotateur actif par lot, y compris sous écritures concurrentes."""

    __tablename__ = "batch_assignments"
    __table_args__ = (
        CheckConstraint(
            "(released_at IS NULL) = (released_by IS NULL)",
            name="liberation_coherente",
        ),
        CheckConstraint(
            "(released_at IS NULL) = (release_reason IS NULL)",
            name="motif_liberation_coherent",
        ),
        CheckConstraint(
            f"release_reason IS NULL OR release_reason IN {_in(RELEASE_REASONS)}",
            name="motif_liberation_valide",
        ),
        Index(
            "uq_batch_assignments_actif",
            "batch_id",
            unique=True,
            postgresql_where=text("released_at IS NULL"),
        ),
        Index("ix_batch_assignments_user", "user_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    batch_id: Mapped[int] = mapped_column(ForeignKey("batches.id", ondelete="CASCADE"))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    assigned_by: Mapped[int] = mapped_column(ForeignKey("users.id"))
    assigned_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    released_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    released_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    release_reason: Mapped[str | None] = mapped_column(Text)


class Image(Base):
    __tablename__ = "images"
    __table_args__ = (
        # L'image et son lot appartiennent à la même session — garanti par la
        # base, sans trigger
        ForeignKeyConstraint(
            ["batch_id", "session_id"],
            ["batches.id", "batches.session_id"],
            name="fk_images_batch_session",
        ),
        CheckConstraint("width > 0 AND height > 0", name="dimensions_positives"),
        # Les 5 champs de crop vont ensemble : tous NULL ou tous renseignés
        CheckConstraint(
            "(cropped_path IS NULL AND crop_x IS NULL AND crop_y IS NULL"
            " AND cropped_width IS NULL AND cropped_height IS NULL)"
            " OR (cropped_path IS NOT NULL AND crop_x IS NOT NULL"
            " AND crop_y IS NOT NULL AND cropped_width IS NOT NULL"
            " AND cropped_height IS NOT NULL)",
            name="crop_complet_ou_absent",
        ),
        CheckConstraint(
            "crop_x IS NULL OR (crop_x >= 0 AND crop_y >= 0"
            " AND cropped_width > 0 AND cropped_height > 0)",
            name="crop_geometrie_valide",
        ),
        CheckConstraint("sha256 ~ '^[0-9a-f]{64}$'", name="sha256_hex"),
        CheckConstraint(f"status IN {_in(IMAGE_STATUSES)}", name="statut_valide"),
        CheckConstraint(
            "status <> 'relue' OR reviewed_by IS NOT NULL", name="relue_a_un_relecteur"
        ),
        # La relecture est le fait d'un AUTRE compte que l'annotateur
        CheckConstraint(
            "reviewed_by IS NULL OR reviewed_by <> annotated_by",
            name="relecteur_different_annotateur",
        ),
        CheckConstraint(
            "(annotated_by IS NULL) = (annotated_at IS NULL)",
            name="annotation_horodatee",
        ),
        CheckConstraint(
            "(reviewed_by IS NULL) = (reviewed_at IS NULL)",
            name="relecture_horodatee",
        ),
        # Une ligne pointée comme remplacée est forcément archivée
        CheckConstraint(
            "superseded_by_id IS NULL OR superseded_at IS NOT NULL",
            name="archivage_coherent",
        ),
        CheckConstraint(
            "preannotation_attempts >= 0",
            name="tentatives_preannotation_positives",
        ),
        CheckConstraint(
            "preannotation_error_kind IS NULL OR preannotation_error_kind IN"
            f" {_in(PREANNOTATION_ERROR_KINDS)}",
            name="erreur_preannotation_valide",
        ),
        CheckConstraint(
            "(preannotation_error IS NULL) = (preannotation_error_kind IS NULL)",
            name="erreur_preannotation_coherente",
        ),
        # Unicité sur les lignes ACTIVES seulement : un recadrage crée une
        # nouvelle ligne qui partage l'original (et donc le sha256) de la
        # ligne archivée. L'import, lui, déduplique contre TOUTES les lignes.
        Index(
            "uq_images_original_path_actif",
            "original_path",
            unique=True,
            postgresql_where=text("superseded_at IS NULL"),
        ),
        Index(
            "uq_images_sha256_actif",
            "sha256",
            unique=True,
            postgresql_where=text("superseded_at IS NULL"),
        ),
        # Deux fichiers d'une même session ne peuvent produire le même .txt
        # à l'export — garanti ici, désambiguïsé à l'import
        Index(
            "uq_images_session_export_actif",
            "session_id",
            "export_filename",
            unique=True,
            postgresql_where=text("superseded_at IS NULL"),
        ),
        # Index complet (non unique) : la recherche de doublons à l'import
        # parcourt aussi les lignes archivées
        Index("ix_images_sha256", "sha256"),
        # Les index de TRAVAIL excluent les lignes archivées : toute requête de
        # file/écran/export doit porter le filtre superseded_at IS NULL — sans
        # lui, l'index partiel est inutilisable et ça se voit au premier EXPLAIN
        Index(
            "ix_images_batch_statut",
            "batch_id",
            "status",
            postgresql_where=text("superseded_at IS NULL"),
        ),
        # Complet : sert aussi l'historique/généalogie par session
        Index("ix_images_session", "session_id"),
        # File du futur worker de pré-annotation
        Index(
            "ix_images_file_preannotation",
            "id",
            postgresql_where=text(
                "status = 'en_attente_preannotation' AND superseded_at IS NULL"
            ),
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"))
    batch_id: Mapped[int] = mapped_column(BigInteger)
    original_filename: Mapped[str] = mapped_column(Text)
    # Nom restitué à l'export, unique par session (3 postes de capture peuvent
    # produire le même IMG_0001.jpg) : préfixé du dossier source UNIQUEMENT en
    # cas de collision réelle ; original_filename garde le nom brut, intact
    export_filename: Mapped[str] = mapped_column(Text)
    # Nom du dossier source (poste de capture) : sert au découpage round-robin
    # des lots — un lot homogène par poste corrélerait le style d'un annotateur
    # à un appareil (biais évitable, décision du 2026-07-31)
    source_label: Mapped[str] = mapped_column(Text)
    # Chemins relatifs à STORAGE_ROOT — jamais absolus
    original_path: Mapped[str] = mapped_column(Text)
    cropped_path: Mapped[str | None] = mapped_column(Text, unique=True)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    # Géométrie du crop en pixels de l'original ; immuable dès qu'une
    # annotation existe (trigger images_gel_crop, SQLSTATE CW001)
    crop_x: Mapped[int | None] = mapped_column(Integer)
    crop_y: Mapped[int | None] = mapped_column(Integer)
    cropped_width: Mapped[int | None] = mapped_column(Integer)
    cropped_height: Mapped[int | None] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(CHAR(64))
    status: Mapped[str] = mapped_column(
        Text, server_default=text("'en_attente_preannotation'")
    )
    # Échec de pré-annotation : l'image reste en `en_attente_preannotation`
    # (pas de statut d'erreur — un incident n'est pas une étape du flux) ;
    # au plafond de tentatives du worker elle est « garée » : hors de la file,
    # comptée à part dans l'avancement des lots. Le succès efface les motifs.
    preannotation_attempts: Mapped[int] = mapped_column(
        Integer, server_default=text("0")
    )
    preannotation_error: Mapped[str | None] = mapped_column(Text)
    preannotation_error_kind: Mapped[str | None] = mapped_column(Text)
    # Dénormalisation de confort — image_status_events fait foi
    annotated_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    annotated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    reviewed_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Archivage par recadrage : superseded_at marque la ligne comme remplacée
    # (c'est lui que testent les index d'unicité partiels), superseded_by_id
    # pointe vers la remplaçante une fois celle-ci insérée
    superseded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    superseded_by_id: Mapped[int | None] = mapped_column(ForeignKey("images.id"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Annotation(Base):
    """Bounding box, coordonnées YOLO normalisées [0,1] relatives au fichier
    annoté : le crop s'il existe, sinon l'original."""

    __tablename__ = "annotations"
    __table_args__ = (
        CheckConstraint("class_id >= 0", name="class_id_positif"),
        CheckConstraint(
            "x_center BETWEEN 0 AND 1 AND y_center BETWEEN 0 AND 1"
            " AND box_width > 0 AND box_width <= 1"
            " AND box_height > 0 AND box_height <= 1",
            name="coordonnees_normalisees",
        ),
        CheckConstraint(f"source IN {_in(ANNOTATION_SOURCES)}", name="source_valide"),
        CheckConstraint(f"state IN {_in(ANNOTATION_STATES)}", name="etat_valide"),
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="confidence_normalisee",
        ),
        CheckConstraint(
            "source <> 'humain' OR confidence IS NULL", name="confidence_modele_seul"
        ),
        CheckConstraint(
            "source <> 'humain' OR model_name IS NULL", name="model_name_modele_seul"
        ),
        CheckConstraint(
            "(source = 'humain') = (created_by IS NOT NULL)",
            name="auteur_humain_coherent",
        ),
        CheckConstraint(
            "(updated_by IS NULL) = (updated_at IS NULL)", name="retouche_horodatee"
        ),
        # Toute validation ET tout rejet ont un auteur ; une 'proposee' n'en a pas
        CheckConstraint(
            "(state = 'proposee') = (decided_by IS NULL)", name="decision_a_un_auteur"
        ),
        CheckConstraint(
            "(decided_by IS NULL) = (decided_at IS NULL)", name="decision_horodatee"
        ),
        Index("ix_annotations_image_state", "image_id", "state"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id", ondelete="CASCADE"))
    class_id: Mapped[int] = mapped_column(SmallInteger)
    x_center: Mapped[float] = mapped_column(Double)
    y_center: Mapped[float] = mapped_column(Double)
    box_width: Mapped[float] = mapped_column(Double)
    box_height: Mapped[float] = mapped_column(Double)
    source: Mapped[str] = mapped_column(Text)
    state: Mapped[str] = mapped_column(Text, server_default=text("'proposee'"))
    confidence: Mapped[float | None] = mapped_column(REAL)
    model_name: Mapped[str | None] = mapped_column(Text)
    # NULL = boîte produite par le worker (source='modele')
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    decided_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ImageStatusEvent(Base):
    """Journal des transitions de statut — source de vérité de la traçabilité
    (les colonnes annotated_*/reviewed_* d'images n'en sont que le reflet).
    Fournira aussi les durées d'annotation (en_cours → annotee)."""

    __tablename__ = "image_status_events"
    __table_args__ = (
        CheckConstraint(
            f"from_status IS NULL OR from_status IN {_in(IMAGE_STATUSES)}",
            name="statut_origine_valide",
        ),
        CheckConstraint(f"to_status IN {_in(IMAGE_STATUSES)}", name="statut_cible_valide"),
        Index("ix_image_status_events_image", "image_id", "changed_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id", ondelete="CASCADE"))
    # NULL à la création de l'image
    from_status: Mapped[str | None] = mapped_column(Text)
    to_status: Mapped[str] = mapped_column(Text)
    # NULL = worker/système
    changed_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    changed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
