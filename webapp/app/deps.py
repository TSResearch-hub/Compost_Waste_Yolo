"""Dépendances FastAPI : session de base, utilisateur courant, rôles."""
from datetime import timedelta
from typing import Annotated

from fastapi import Cookie, Depends, HTTPException

from .config import get_settings
from .db import get_sessionmaker
from .models import AuthSession, User
from .security import hash_token, utcnow

SESSION_COOKIE = "compost_session"

_NON_AUTHENTIFIE = HTTPException(status_code=401, detail="Non authentifié")


def get_db():
    session = get_sessionmaker()()
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


def get_current_user(
    db=Depends(get_db),
    token: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None,
) -> User:
    if not token:
        raise _NON_AUTHENTIFIE

    auth = db.get(AuthSession, hash_token(token))
    if auth is None:
        raise _NON_AUTHENTIFIE

    # Expiration glissante par inactivité
    cutoff = utcnow() - timedelta(minutes=get_settings().auth_inactivity_minutes)
    if auth.last_seen_at < cutoff:
        db.delete(auth)
        db.commit()  # la purge doit survivre au rollback déclenché par le 401
        raise _NON_AUTHENTIFIE

    user = db.get(User, auth.user_id)
    if user is None or not user.is_active:
        db.delete(auth)
        db.commit()
        raise _NON_AUTHENTIFIE

    auth.last_seen_at = utcnow()  # renouvellement — commité en fin de requête
    return user


def require_roles(*roles: str):
    def dependency(user: User = Depends(get_current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(status_code=403, detail="Droits insuffisants")
        return user

    return dependency
