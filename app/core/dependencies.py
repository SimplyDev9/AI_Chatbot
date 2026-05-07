from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt
from sqlalchemy.orm import Session

from app.core.security import SECRET_KEY, ALGORITHM
from app.db.database import get_db
from app.db.models import User

security = HTTPBearer()


def get_current_user(
        credentials: HTTPAuthorizationCredentials = Depends(security),
        db: Session = Depends(get_db)
):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        email = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token")

        user = db.query(User).filter(
            User.email == email,
            User.is_active == True
        ).first()
        if not user:
            raise HTTPException(status_code=401, detail="User not found or inactive")

        return user

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def require_permission(permission: str):
    """permission must be lowercase, matching seed.py: 'chat', 'ingest', 'manage_users'"""
    def checker(user: User = Depends(get_current_user)):
        user_permissions = {
            perm.name
            for role in user.roles
            for perm in role.permissions
        }

        if permission not in user_permissions:
            raise HTTPException(
                status_code=403,
                detail=f"Permission '{permission}' required"
            )

        return user

    return checker