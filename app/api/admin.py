from app.db.seed import seed_roles_permissions
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.db.database import get_db
from app.db.models import Role, Permission, User
from app.core.dependencies import require_permission

router = APIRouter()

# ------------------------
# REQUEST MODELS
# ------------------------

class CreateRoleRequest(BaseModel):
    role_name: str


class AssignRoleRequest(BaseModel):
    email: str
    role_name: str


class UpdateRolePermissionsRequest(BaseModel):
    role_name: str
    permissions: list[str]


class RemoveRoleRequest(BaseModel):
    email: str
    role_name: str


class DeleteRoleRequest(BaseModel):
    role_name: str


class RemovePermissionRequest(BaseModel):
    role_name: str
    permission_name: str


class DeleteUserRequest(BaseModel):
    email: str


class ReactivateUserRequest(BaseModel):
    email: str


# ------------------------
# SEED
# ------------------------
@router.get("/seed")
def seed_data(db: Session = Depends(get_db)):
    return {"message": seed_roles_permissions(db)}


# ------------------------
# CREATE ROLE
# ------------------------
@router.post("/admin/create-role")
def create_role(
        req: CreateRoleRequest,
        db: Session = Depends(get_db),
        user=Depends(require_permission("manage_users"))
):
    existing = db.query(Role).filter(Role.name == req.role_name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Role already exists")

    role = Role(name=req.role_name)
    db.add(role)
    db.commit()

    return {"message": f"Role '{req.role_name}' created"}


# ------------------------
# ASSIGN ROLE
# ------------------------
@router.post("/admin/assign-role")
def assign_role(
        req: AssignRoleRequest,
        db: Session = Depends(get_db),
        user=Depends(require_permission("manage_users"))
):
    user_obj = db.query(User).filter(
        User.email == req.email,
        User.is_active == True
    ).first()

    if not user_obj:
        raise HTTPException(status_code=404, detail="Active user not found")

    role = db.query(Role).filter(Role.name == req.role_name).first()
    if not role:
        raise HTTPException(status_code=404, detail="Role not found")

    if role in user_obj.roles:
        return {"message": "Role already assigned"}

    user_obj.roles.append(role)
    db.commit()

    return {"message": f"Role '{req.role_name}' assigned to {req.email}"}


# ------------------------
# UPDATE ROLE PERMISSIONS
# ------------------------
@router.post("/admin/update-role-permissions")
def update_role_permissions(
        req: UpdateRolePermissionsRequest,
        db: Session = Depends(get_db),
        user=Depends(require_permission("manage_users"))
):
    role = db.query(Role).filter(Role.name == req.role_name).first()
    if not role:
        raise HTTPException(status_code=404, detail="Role not found")

    permission_objs = db.query(Permission).filter(
        Permission.name.in_(req.permissions)
    ).all()

    role.permissions = permission_objs
    db.commit()

    return {
        "message": f"Permissions updated for role '{req.role_name}'",
        "permissions": req.permissions
    }


# ------------------------
# DELETE ROLE
# ------------------------
@router.delete("/admin/delete-role")
def delete_role(
        req: DeleteRoleRequest,
        db: Session = Depends(get_db),
        user=Depends(require_permission("manage_users"))
):
    role = db.query(Role).filter(Role.name == req.role_name).first()

    if not role:
        raise HTTPException(status_code=404, detail="Role not found")

    role.permissions.clear()
    role.users.clear()

    db.delete(role)
    db.commit()

    return {"message": f"Role '{req.role_name}' deleted"}


# ------------------------
# REMOVE ROLE FROM USER
# ------------------------
@router.delete("/admin/remove-role")
def remove_role_from_user(
        req: RemoveRoleRequest,
        db: Session = Depends(get_db),
        user=Depends(require_permission("manage_users"))
):
    user_obj = db.query(User).filter(
        User.email == req.email,
        User.is_active == True
    ).first()

    if not user_obj:
        raise HTTPException(status_code=404, detail="Active user not found")

    role = db.query(Role).filter(Role.name == req.role_name).first()
    if not role:
        raise HTTPException(status_code=404, detail="Role not found")

    if role not in user_obj.roles:
        return {"message": "User does not have this role"}

    user_obj.roles.remove(role)
    db.commit()

    return {"message": f"Role '{req.role_name}' removed from {req.email}"}


# ------------------------
# REMOVE PERMISSION FROM ROLE
# ------------------------
@router.delete("/admin/remove-permission-from-role")
def remove_permission_from_role(
        req: RemovePermissionRequest,
        db: Session = Depends(get_db),
        user=Depends(require_permission("manage_users"))
):
    role = db.query(Role).filter(Role.name == req.role_name).first()
    if not role:
        raise HTTPException(status_code=404, detail="Role not found")

    permission = db.query(Permission).filter(
        Permission.name == req.permission_name
    ).first()

    if not permission:
        raise HTTPException(status_code=404, detail="Permission not found")

    if permission not in role.permissions:
        return {"message": "Permission not assigned to role"}

    role.permissions.remove(permission)
    db.commit()

    return {
        "message": f"Permission '{req.permission_name}' removed from role '{req.role_name}'"
    }


# ------------------------
# SOFT DELETE USER
# ------------------------
@router.delete("/admin/delete-user")
def delete_user(
        req: DeleteUserRequest,
        db: Session = Depends(get_db),
        current_user=Depends(require_permission("manage_users"))
):
    user = db.query(User).filter(User.email == req.email).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.email == current_user.email:
        raise HTTPException(status_code=400, detail="You cannot delete yourself")

    admin_role = db.query(Role).filter(Role.name == "ADMIN").first()

    if admin_role and admin_role in user.roles:
        admin_count = (
            db.query(User)
            .join(User.roles)
            .filter(Role.name == "ADMIN", User.is_active == True)
            .count()
        )

        if admin_count <= 1:
            raise HTTPException(
                status_code=400,
                detail="Cannot delete the last admin user"
            )

    user.is_active = False
    user.roles.clear()

    db.commit()

    return {"message": f"User '{req.email}' deactivated successfully"}


# ------------------------
# LIST USERS
# ------------------------
@router.get("/admin/list-users")
def list_users(
        db: Session = Depends(get_db),
        user=Depends(require_permission("manage_users"))
):
    users = db.query(User).filter(User.is_active == True).all()

    return [
        {
            "email": u.email,
            "roles": [r.name for r in u.roles]
        }
        for u in users
    ]


# # ------------------------
# # REACTIVATE USER
# # ------------------------
# @router.post("/admin/reactivate-user")
# def reactivate_user(
#         req: ReactivateUserRequest,
#         db: Session = Depends(get_db),
#         user=Depends(require_permission("manage_users"))
# ):
#     user_obj = db.query(User).filter(User.email == req.email).first()
#
#     if not user_obj:
#         raise HTTPException(status_code=404, detail="User not found")
#
#     user_obj.is_active = True
#     db.commit()
#
#     return {"message": f"User '{req.email}' reactivated"}