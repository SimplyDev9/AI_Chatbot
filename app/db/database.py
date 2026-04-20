from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

from app.core.config import DATABASE_URL

# ✅ CREATE ENGINE
engine = create_engine(DATABASE_URL)

# ✅ SESSION
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()


# ✅ Test connection function (keep this)
def test_connection():
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            return "DB Connected Successfully ✅"
    except Exception as e:
        return f"DB Connection Failed ❌: {str(e)}"

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()