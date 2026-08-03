"""Échec de pré-annotation (lot L2 — worker).

Une image dont l'inférence échoue reste en `en_attente_preannotation` : pas de
statut d'erreur (un incident n'est pas une étape du flux, la liste blanche des
transitions est intacte). L'échec est porté par trois colonnes : compteur de
tentatives, motif libre, catégorie grossière requêtable. Au plafond de
tentatives du worker, l'image est « garée » : exclue de la file, comptée à
part dans l'avancement des lots (GET /api/batches).

Première migration incrémentale : la base de dev est provisionnée, on
n'amende plus 0001.

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-31

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0002'
down_revision: Union[str, None] = '0001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('images', sa.Column(
        'preannotation_attempts', sa.Integer(),
        server_default=sa.text('0'), nullable=False))
    op.add_column('images', sa.Column('preannotation_error', sa.Text(), nullable=True))
    op.add_column('images', sa.Column('preannotation_error_kind', sa.Text(), nullable=True))
    op.create_check_constraint(
        op.f('ck_images_tentatives_preannotation_positives'), 'images',
        'preannotation_attempts >= 0')
    op.create_check_constraint(
        op.f('ck_images_erreur_preannotation_valide'), 'images',
        "preannotation_error_kind IS NULL OR preannotation_error_kind IN"
        " ('fichier_illisible', 'moteur_indisponible', 'invariant_viole')")
    op.create_check_constraint(
        op.f('ck_images_erreur_preannotation_coherente'), 'images',
        '(preannotation_error IS NULL) = (preannotation_error_kind IS NULL)')


def downgrade() -> None:
    op.drop_constraint(op.f('ck_images_erreur_preannotation_coherente'), 'images')
    op.drop_constraint(op.f('ck_images_erreur_preannotation_valide'), 'images')
    op.drop_constraint(op.f('ck_images_tentatives_preannotation_positives'), 'images')
    op.drop_column('images', 'preannotation_error_kind')
    op.drop_column('images', 'preannotation_error')
    op.drop_column('images', 'preannotation_attempts')
