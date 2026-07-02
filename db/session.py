import os
from contextlib import contextmanager
from typing import Generator, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

_engine = None
_SessionLocal: Optional[sessionmaker] = None



def get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        db_url = os.environ["DATABASE_URL"]
        _engine = create_engine(
            db_url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
        _SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False)
        logger.info("Database engine created")
    return _engine


@contextmanager
def get_session() -> Generator[Session, None, None]:
    engine = get_engine()
    assert _SessionLocal is not None
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db():
    from db.models import Base
    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("All tables created / verified")


def health_check() -> bool:
    try:
        with get_session() as s:
            s.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"DB health check failed: {e}")
        return False
