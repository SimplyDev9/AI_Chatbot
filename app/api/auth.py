from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.db.database import get_db
from app.db.models import User, Role
from app.core.security import hash_password, verify_password, create_access_token

router = APIRouter(prefix="/auth", tags=["Auth"])


class LoginRequest(BaseModel):
    email: str
    password: str


class SignupRequest(BaseModel):
    email: str
    password: str
    role_name: str = "USER"

# ✅ SIGNUP
@router.post("/signup")
def signup(req: SignupRequest, db: Session = Depends(get_db)):

    existing = db.query(User).filter(User.email == req.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")

    role = db.query(Role).filter(Role.name == req.role_name).first()
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


# ✅ LOGIN
@router.post("/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()

    if not user or not verify_password(req.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    roles = [role.name for role in user.roles]

    permissions = {
        perm.name
        for role in user.roles
        for perm in role.permissions
    }

    token = create_access_token({
        "sub": user.email,
        "roles": roles
    })

    return {
        "access_token": token,
        "token_type": "bearer",
        "roles": roles,
        "permissions": list(permissions)
    }