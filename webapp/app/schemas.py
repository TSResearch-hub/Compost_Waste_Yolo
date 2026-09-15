"""Schémas Pydantic des entrées/sorties API."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, computed_field, field_validator

from .models import JETSON_ID_REGEX, ROLES, normaliser_jetson_id
from .security import PASSWORD_MIN_LENGTH

Role = Literal[*ROLES]


class LoginIn(BaseModel):
    username: str = Field(min_length=1)
    password: str = Field(min_length=1)


class UserOut(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    username: str
    display_name: str | None
    role: str
    is_active: bool
    # vrai = mot de passe posé par un administrateur, à remplacer à la
    # prochaine connexion (POST /api/auth/changer-mot-de-passe)
    must_change_password: bool
    created_at: datetime


class UserCreate(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=PASSWORD_MIN_LENGTH)
    display_name: str | None = None
    role: Role


class UserPatch(BaseModel):
    """Champs modifiables par un administrateur. Tous optionnels —
    `display_name` ne compte que s'il est présent dans le corps (null
    explicite = effacer le nom affiché)."""

    is_active: bool | None = None
    role: Role | None = None
    display_name: str | None = None
    password: str | None = Field(default=None, min_length=PASSWORD_MIN_LENGTH)


class ChangementMotDePasseIn(BaseModel):
    """Changement de SON PROPRE mot de passe : l'actuel est exigé — un poste
    laissé déverrouillé ne suffit pas à voler le compte."""

    actuel: str = Field(min_length=1)
    nouveau: str = Field(min_length=PASSWORD_MIN_LENGTH)


# ── Flotte Jetson ─────────────────────────────────────────────────────────────

class JetsonOut(BaseModel):
    model_config = {"from_attributes": True}

    id: str
    name: str | None
    is_active: bool
    created_at: datetime
    updated_at: datetime


class JetsonCreate(BaseModel):
    """Déclaration d'une carte par un administrateur — préalable à tout envoi.
    L'identifiant est normalisé (minuscules, `:` → `-`) AVANT validation :
    `48:B0:2D:3E:AA:01` et `48-b0-2d-3e-aa-01` désignent la même carte."""

    id: str = Field(pattern=JETSON_ID_REGEX)
    name: str | None = None

    @field_validator("id", mode="before")
    @classmethod
    def _normaliser(cls, v):
        return normaliser_jetson_id(v) if isinstance(v, str) else v


class JetsonPatch(BaseModel):
    """Champs modifiables par un administrateur. Tous optionnels — `name` ne
    compte que s'il est présent dans le corps (null explicite = effacer le
    nom). `is_active` faux : la carte est refusée à l'envoi, ses sessions
    et images restent."""

    name: str | None = None
    is_active: bool | None = None


# ── Distribution des modèles IA ───────────────────────────────────────────────

class ModelVersionOut(BaseModel):
    """Métadonnées d'une version publiée. `pt_url` / `yaml_url` sont les
    chemins à appeler (même origine que l'API) avec le jeton SYNC_TOKEN en
    `Authorization: Bearer …` pour télécharger les fichiers ; les
    `*_file_path` sont les chemins internes au stockage, donnés pour
    information. La publication (POST /api/models_ia/upload) est un
    formulaire multipart (`version_name`, `pt_file`, `yaml_file`) — pas de
    schéma d'entrée JSON."""

    model_config = {"from_attributes": True}

    id: int
    version_name: str
    pt_file_path: str
    yaml_file_path: str
    created_by: int
    created_at: datetime

    @computed_field
    @property
    def pt_url(self) -> str:
        return f"/api/models_ia/{self.id}/fichier/pt"

    @computed_field
    @property
    def yaml_url(self) -> str:
        return f"/api/models_ia/{self.id}/fichier/yaml"
