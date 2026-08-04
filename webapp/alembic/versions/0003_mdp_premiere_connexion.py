"""Changement de mot de passe obligatoire à la première connexion.

`users.must_change_password` : posé à vrai quand un administrateur fixe le
mot de passe (création du compte, réinitialisation) — l'intéressé le remplace
à sa prochaine connexion via POST /api/auth/changer-mot-de-passe. Les comptes
existants restent à faux : leurs mots de passe sont déjà en usage.

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-04

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0003'
down_revision: Union[str, None] = '0002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('users', sa.Column(
        'must_change_password', sa.Boolean(),
        server_default=sa.text('false'), nullable=False))


def downgrade() -> None:
    op.drop_column('users', 'must_change_password')
