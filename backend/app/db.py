import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return False

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DB_URL")

if DATABASE_URL:
    engine = create_engine(DATABASE_URL)

# Create Session
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base class for all database models
Base = declarative_base()


# Dependency to get database session
def get_db():
    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()