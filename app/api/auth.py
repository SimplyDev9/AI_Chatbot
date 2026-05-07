from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from starlette.requests import Request

from app.db.database import get_db
from app.db.models import User, Role
from app.db.seed import seed_roles_permissions
from app.core.security import hash_password, verify_password, create_access_token
from app.core.dependencies import get_current_user
from app.core.limiter import limiter

router = APIRouter(prefix="/auth", tags=["Auth"])


class LoginRequest(BaseModel):
    email: str
    password: str


class SignupRequest(BaseModel):
    email: str
    password: str
    role_name: str = "USER"


class FirstSetupRequest(BaseModel):
    email: str
    password: str


@router.post("/first-setup")
def first_setup(req: FirstSetupRequest, db: Session = Depends(get_db)):
    """
    One-time bootstrap endpoint for a fresh database.
    Disabled automatically once any user exists.
    """
    if db.query(User).first():
        raise HTTPException(
            status_code=403,
            detail="Setup already completed. This endpoint is disabled once users exist."
        )

    # Use the existing seed.py — single source of truth for roles/permissions
    seed_roles_permissions(db)

    # Fetch the ADMIN role that seed just created
    admin_role = db.query(Role).filter(Role.name == "ADMIN").first()

    # Create the first admin user
    admin = User(
        email=req.email,
        password=hash_password(req.password),
        is_active=True,
    )
    admin.roles.append(admin_role)
    db.add(admin)
    db.commit()

    token = create_access_token({"sub": admin.email, "roles": ["ADMIN"]})

    return {
        "message": "Setup complete. Roles, permissions, and first admin created.",
        "email": admin.email,
        "roles": ["ADMIN"],
        "permissions": ["chat", "ingest", "manage_users"],
        "access_token": token,
        "token_type": "bearer",
    }


@router.post("/signup")
@limiter.limit("3/minute")
def signup(request: Request, req: SignupRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == req.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")

    role = db.query(Role).filter(Role.name == req.role_name.strip().upper()).first()
    if not role:
        raise HTTPException(status_code=400, detail="Invalid role")

    new_user = User(
        email=req.email,
        password=hash_password(req.password)
    )
    new_user.roles.append(role)
    db.add(new_user)
    db.commit()

    return {"message": "User created successfully"}


@router.post("/login")
@limiter.limit("5/minute")
def login(request: Request, req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email, User.is_active == True).first()

    if not user or not verify_password(req.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    roles = [role.name for role in user.roles]

    permissions = {
        perm.name
        for role in user.roles
        for perm in role.permissions
    }

    token = create_access_token({"sub": user.email, "roles": roles})

    return {
        "access_token": token,
        "token_type": "bearer",
        "roles": roles,
        "permissions": list(permissions)
    }

@router.get("/me")
def get_me(user: User = Depends(get_current_user)):
    """Returns current user permissions fresh from DB — never trust client storage."""
    permissions = {
        perm.name
        for role in user.roles
        for perm in role.permissions
    }
    return {
        "email": user.email,
        "roles": [r.name for r in user.roles],
        "permissions": list(permissions),
        "is_active": user.is_active,
    }