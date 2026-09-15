"""ajout_distribution_modeles

Distribution des poids du modèle IA aux cartes Jetson : `model_versions`, une
ligne par version publiée par un administrateur (POST /api/models_ia/upload),
avec le nom de version (unique, même alphabet que les identifiants de carte —
il nomme le dossier `models/{version_name}/` du stockage, CHECK
`nom_version_valide`), les chemins relatifs du `.pt` et du `data.yaml`,
l'auteur et la date. Les cartes lisent GET /api/models_ia/latest (jeton
SYNC_TOKEN) : la plus récente par created_at. Jamais modifiée ni supprimée.

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-15 10:24:05.802850

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0006'
down_revision: Union[str, None] = '0005'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('model_versions',
    sa.Column('id', sa.BigInteger(), sa.Identity(always=True), nullable=False),
    sa.Column('version_name', sa.Text(), nullable=False),
    sa.Column('pt_file_path', sa.Text(), nullable=False),
    sa.Column('yaml_file_path', sa.Text(), nullable=False),
    sa.Column('created_by', sa.BigInteger(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("version_name ~ '^[A-Za-z0-9_.-]{1,100}$'", name=op.f('ck_model_versions_nom_version_valide')),
    sa.ForeignKeyConstraint(['created_by'], ['users.id'], name=op.f('fk_model_versions_created_by_users')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_model_versions')),
    sa.UniqueConstraint('version_name', name=op.f('uq_model_versions_version_name'))
    )


def downgrade() -> None:
    op.drop_table('model_versions')
