"""Schémas Pydantic des entrées/sorties API."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from .models import ROLES
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
    created_at: datetime


class UserCreate(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=PASSWORD_MIN_LENGTH)
    display_name: str | None = None
    role: Role


class UserPatch(BaseModel):
    """Champs modifiables par un administrateur. Tous optionnels."""

    is_active: bool | None = None
    role: Role | None = None
    password: str | None = Field(default=None, min_length=PASSWORD_MIN_LENGTH)
