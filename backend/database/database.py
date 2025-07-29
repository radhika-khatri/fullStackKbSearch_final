from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import NullPool
import os
from dotenv import load_dotenv
load_dotenv()

# Update with your local DB credentials
DATABASE_URL = os.getenv("DATABASE_URL")

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=True,
    future=True,
    poolclass=NullPool  # good for dev to avoid connection leaks
)

# Async session factory
async_session = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Base model class for ORM models
Base = declarative_base() 
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

async def get_db():
    """
    FastAPI dependency to get an async DB session.
    Usage in path operation:
        async def endpoint(db: AsyncSession = Depends(get_db)):
            ...
    """
    # db = SessionLocal()
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()

# async def get_db():
#     from models import db_models
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

async def init_db():
    """
    Initialize database: create all tables for ORM models.
    Call this on FastAPI startup.
    """
    # import all modules that define models so they are registered on the Base metadata
      # noqa: F401

    async with engine.begin() as conn:
        # run_sync will execute the synchronous create_all on the async connection
        await conn.run_sync(Base.metadata.create_all)
