"""Connexion / déconnexion / session courante."""
from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Cookie, Depends, HTTPException, Response
from sqlalchemy import delete, func, select

from ..config import get_settings
from ..deps import SESSION_COOKIE, get_current_user, get_db
from ..models import AuthSession, User
from ..schemas import LoginIn, UserOut
from ..security import hash_token, new_session_token, utcnow, verify_password

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login", response_model=UserOut)
def login(body: LoginIn, response: Response, db=Depends(get_db)):
    user = db.scalar(
        select(User).where(func.lower(User.username) == body.username.lower())
    )
    # Message unique quel que soit l'échec : ne révèle pas l'existence du compte
    if user is None or not user.is_active or not verify_password(
        user.password_hash, body.password
    ):
        raise HTTPException(status_code=401, detail="Identifiants invalides")

    # Purge opportuniste des sessions expirées (toutes, pas que les siennes)
    cutoff = utcnow() - timedelta(minutes=get_settings().auth_inactivity_minutes)
    db.execute(delete(AuthSession).where(AuthSession.last_seen_at < cutoff))

    token = new_session_token()
    db.add(AuthSession(token_hash=hash_token(token), user_id=user.id))
    # Cookie de session (pas de max_age : l'expiration par inactivité est
    # tenue côté serveur). `secure` viendra avec le TLS du déploiement.
    response.set_cookie(SESSION_COOKIE, token, httponly=True, samesite="lax")
    return user


@router.post("/logout", status_code=204)
def logout(
    response: Response,
    db=Depends(get_db),
    token: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None,
):
    if token:
        auth = db.get(AuthSession, hash_token(token))
        if auth is not None:
            db.delete(auth)
    response.delete_cookie(SESSION_COOKIE)


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return user
