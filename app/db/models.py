from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, Table
from sqlalchemy.orm import relationship

from app.db.database import Base


# ─────────────────────────────────────────────────────────────────────────────
# Association tables
#
# Each join table carries the human-readable columns (email / role_name /
# permission_name) alongside the FK integers.  This makes every table directly
# readable in pgAdmin or psql without needing joins or views.
# ─────────────────────────────────────────────────────────────────────────────

user_roles = Table(
    "user_roles",
    Base.metadata,
    Column("user_id",   Integer, ForeignKey("users.id",       ondelete="CASCADE"), primary_key=True),
    Column("role_id",   Integer, ForeignKey("roles.id",        ondelete="CASCADE"), primary_key=True),
    # ↓ human-readable denormalised columns — kept in sync by the ORM event below
    Column("user_email",  String(255), nullable=False, index=True),
    Column("role_name",   String(100), nullable=False, index=True),
)

role_permissions = Table(
    "role_permissions",
    Base.metadata,
    Column("role_id",       Integer, ForeignKey("roles.id",       ondelete="CASCADE"), primary_key=True),
    Column("permission_id", Integer, ForeignKey("permissions.id", ondelete="CASCADE"), primary_key=True),
    # ↓ human-readable denormalised columns
    Column("role_name",       String(100), nullable=False, index=True),
    Column("permission_name", String(100), nullable=False, index=True),
)


# ─────────────────────────────────────────────────────────────────────────────
# ORM models
# ─────────────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id        = Column(Integer, primary_key=True, index=True)
    email     = Column(String(255), unique=True, index=True, nullable=False)
    password  = Column(String,      nullable=False)
    is_active = Column(Boolean,     default=True,  nullable=False)

    roles = relationship("Role", secondary=user_roles, back_populates="users")

    def __repr__(self):
        return f"<User email={self.email!r} active={self.is_active}>"


class Role(Base):
    __tablename__ = "roles"

    id   = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)

    users       = relationship("User",       secondary=user_roles,       back_populates="roles")
    permissions = relationship("Permission", secondary=role_permissions,  back_populates="roles")

    def __repr__(self):
        return f"<Role name={self.name!r}>"


class Permission(Base):
    __tablename__ = "permissions"

    id   = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)

    roles = relationship("Role", secondary=role_permissions, back_populates="permissions")

    def __repr__(self):
        return f"<Permission name={self.name!r}>"


# ─────────────────────────────────────────────────────────────────────────────
# SQLAlchemy event listeners — keep denormalised columns in sync automatically
#
# These fire whenever the ORM appends/removes from a Many-to-Many collection
# so you never have to remember to update them manually.
# ─────────────────────────────────────────────────────────────────────────────

from sqlalchemy import event
from sqlalchemy.orm import Session as _Session


def _sync_user_roles_insert(target_session: _Session, flush_context):
    """
    Before flush: for every pending user_roles insert, fill user_email and role_name.
    SQLAlchemy doesn't expose M2M inserts as ORM events directly, so we hook
    into after_bulk_insert on the association table via the engine-level event.
    """
    pass  # handled by after_flush below


@event.listens_for(_Session, "after_bulk_delete")
def _after_bulk_delete(delete_context):
    pass  # cascade handles FK cleanup


# Simpler approach: patch the relationship append/remove at the instance level.
# When role is appended to user.roles, we directly insert with readable columns.

from sqlalchemy import event as _sa_event


@_sa_event.listens_for(User.roles, "append")
def _user_role_append(user: User, role: Role, initiator):
    """Fires when admin does: user.roles.append(role)"""
    # The actual table row is inserted by SQLAlchemy; we update it post-flush
    # via after_flush_postexec to set the readable columns.
    if not hasattr(user, "_pending_role_appends"):
        user._pending_role_appends = []
    user._pending_role_appends.append(role)


@_sa_event.listens_for(Role.permissions, "append")
def _role_permission_append(role: Role, permission, initiator):
    if not hasattr(role, "_pending_perm_appends"):
        role._pending_perm_appends = []
    role._pending_perm_appends.append(permission)


@_sa_event.listens_for(_Session, "after_flush_postexec")
def _after_flush_postexec(session: _Session, flush_context):
    """
    After SQLAlchemy has written the M2M rows, backfill the readable columns.
    We use raw UPDATE so SQLAlchemy doesn't loop infinitely.
    """
    from sqlalchemy import text

    conn = session.connection()

    # Sync user_roles readable columns
    conn.execute(text("""
        UPDATE user_roles ur
        SET    user_email = u.email,
               role_name  = r.name
        FROM   users u,
               roles r
        WHERE  ur.user_id  = u.id
        AND    ur.role_id   = r.id
        AND    (ur.user_email IS NULL OR ur.user_email = ''
             OR ur.role_name  IS NULL OR ur.role_name  = '')
    """))

    # Sync role_permissions readable columns
    conn.execute(text("""
        UPDATE role_permissions rp
        SET    role_name       = r.name,
               permission_name = p.name
        FROM   roles r,
               permissions p
        WHERE  rp.role_id       = r.id
        AND    rp.permission_id = p.id
        AND    (rp.role_name       IS NULL OR rp.role_name       = ''
             OR rp.permission_name IS NULL OR rp.permission_name = '')
    """))