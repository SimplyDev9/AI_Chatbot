from sqlalchemy.orm import Session

from app.db.models import Role, Permission


def seed_roles_permissions(db: Session):
    if db.query(Role).first():
        return "Already Seeded"

    # Permissions — all lowercase, single source of truth
    chat         = Permission(name="chat")
    ingest       = Permission(name="ingest")
    manage_users = Permission(name="manage_users")

    db.add_all([chat, ingest, manage_users])
    db.commit()

    # Roles
    admin_role = Role(name="ADMIN")
    user_role  = Role(name="USER")

    admin_role.permissions = [chat, ingest, manage_users]
    user_role.permissions  = [chat]

    db.add_all([admin_role, user_role])
    db.commit()

    return "Seeding Done ✅"