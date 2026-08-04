"""Relecture : la transition en_cours → relue entre dans la liste blanche.

Ouvrir est le seul mode d'accès à une image (elle passe `en_cours`) : valider
ou corriger une relecture part donc toujours de `en_cours`. La paire sert
aussi à la fermeture (POST /images/{id}/fermer) et à la libération d'un lot
pour restituer une image `relue` rouverte puis abandonnée. Le trigger CW002
est remplacé À L'IDENTIQUE de models.IMAGE_STATUS_TRANSITIONS — toute
évolution doit toucher les deux (test_transitions_whitelist les compare).

Revision ID: 0004
Revises: 0003
Create Date: 2026-08-04

"""
from typing import Sequence, Union

from alembic import op


revision: str = '0004'
down_revision: Union[str, None] = '0003'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_FONCTION = """
    CREATE OR REPLACE FUNCTION trg_images_transition_statut() RETURNS trigger
    LANGUAGE plpgsql AS $$
    BEGIN
        IF NEW.status IS DISTINCT FROM OLD.status
           AND (OLD.status, NEW.status) NOT IN ({paires}) THEN
            RAISE EXCEPTION
                'transition de statut interdite : % -> % (image %)',
                OLD.status, NEW.status, OLD.id
                USING ERRCODE = 'CW002';
        END IF;
        RETURN NEW;
    END $$;
"""

_AVEC_RELECTURE = """
    ('en_attente_preannotation', 'a_annoter'),
    ('a_annoter', 'en_cours'),
    ('en_cours', 'a_annoter'),
    ('en_cours', 'annotee'),
    ('en_cours', 'relue'),
    ('annotee', 'relue'),
    ('annotee', 'en_cours'),
    ('relue', 'en_cours')
"""

_SANS_RELECTURE = """
    ('en_attente_preannotation', 'a_annoter'),
    ('a_annoter', 'en_cours'),
    ('en_cours', 'a_annoter'),
    ('en_cours', 'annotee'),
    ('annotee', 'relue'),
    ('annotee', 'en_cours'),
    ('relue', 'en_cours')
"""


def upgrade() -> None:
    op.execute(_FONCTION.format(paires=_AVEC_RELECTURE))


def downgrade() -> None:
    op.execute(_FONCTION.format(paires=_SANS_RELECTURE))
