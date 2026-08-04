"""Gestion des comptes — administrateur uniquement, pas d'auto-inscription."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from ..deps import get_db, require_roles
from ..models import User
from ..schemas import UserCreate, UserOut, UserPatch
from ..security import hash_password

router = APIRouter(
    prefix="/api/users",
    tags=["users"],
    dependencies=[Depends(require_roles("administrateur"))],
)


@router.post("", response_model=UserOut, status_code=201)
def create_user(
    body: UserCreate,
    db=Depends(get_db),
    admin: User = Depends(require_roles("administrateur")),
):
    user = User(
        username=body.username,
        password_hash=hash_password(body.password),
        display_name=body.display_name,
        role=body.role,
        # mot de passe initial posé par l'admin : à remplacer à la première
        # connexion
        must_change_password=True,
        created_by=admin.id,
    )
    db.add(user)
    try:
        db.flush()
    except IntegrityError:
        raise HTTPException(status_code=409, detail="Nom d'utilisateur déjà pris")
    return user


@router.get("", response_model=list[UserOut])
def list_users(db=Depends(get_db)):
    return db.scalars(select(User).order_by(User.id)).all()


@router.patch("/{user_id}", response_model=UserOut)
def patch_user(
    user_id: int,
    body: UserPatch,
    db=Depends(get_db),
    admin: User = Depends(require_roles("administrateur")),
):
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="Utilisateur inconnu")
    # Anti-verrouillage : un admin ne modifie ni son propre rôle ni son statut
    if user.id == admin.id and (body.is_active is not None or body.role is not None):
        raise HTTPException(
            status_code=400,
            detail="Un administrateur ne modifie pas son propre rôle ou statut",
        )
    if body.is_active is not None:
        user.is_active = body.is_active
    if body.role is not None:
        user.role = body.role
    if "display_name" in body.model_fields_set:  # null explicite = effacer
        user.display_name = body.display_name
    if body.password is not None:
        user.password_hash = hash_password(body.password)
        # mot de passe posé pour AUTRUI : l'intéressé le remplace à sa
        # prochaine connexion ; le sien, l'admin vient de le choisir
        user.must_change_password = user.id != admin.id
    return user
