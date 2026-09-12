"""Flotte Jetson : registre des cartes et origine des sessions.

`jetson_devices` : une ligne par carte déclarée par un administrateur
(identifiant = adresse MAC normalisée ou tout identifiant stable — voir
models.normaliser_jetson_id, même forme exigée par le CHECK
`identifiant_valide`), avec un interrupteur `is_active` : POST /api/sync/upload
refuse toute carte inconnue ou désactivée. Pas de suppression : désactiver
retire la carte du service en conservant ses sessions.

`sessions.jetson_id` (nullable, FK, indexé) relie une session à la carte qui
l'a OUVERTE par envoi automatique ; NULL pour les imports manuels (CLI, écran
Technique) ; un rattachement ultérieur ne le modifie pas.

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-12 06:33:27.684781

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0005'
down_revision: Union[str, None] = '0004'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('jetson_devices',
    sa.Column('id', sa.Text(), nullable=False),
    sa.Column('name', sa.Text(), nullable=True),
    sa.Column('is_active', sa.Boolean(), server_default=sa.text('true'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("id ~ '^[a-z0-9_.-]{1,100}$'", name=op.f('ck_jetson_devices_identifiant_valide')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_jetson_devices'))
    )
    op.add_column('sessions', sa.Column('jetson_id', sa.Text(), nullable=True))
    op.create_foreign_key(
        op.f('fk_sessions_jetson_id_jetson_devices'), 'sessions', 'jetson_devices',
        ['jetson_id'], ['id'])
    op.create_index('ix_sessions_jetson', 'sessions', ['jetson_id'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_sessions_jetson', table_name='sessions')
    op.drop_constraint(op.f('fk_sessions_jetson_id_jetson_devices'), 'sessions',
                       type_='foreignkey')
    op.drop_column('sessions', 'jetson_id')
    op.drop_table('jetson_devices')
