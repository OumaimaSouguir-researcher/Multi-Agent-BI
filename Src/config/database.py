from sqlalchemy import create_engine, event
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.pool import QueuePool
import logging

from .settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Create declarative base for models
Base = declarative_base()


class DatabaseConfig:
    """Database configuration and connection management"""
    
    def __init__(self):
        self.settings = settings
        
        # Sync engine (for scripts and migrations)
        self.engine = create_engine(
            self.settings.database_url,
            poolclass=QueuePool,
            pool_size=self.settings.POSTGRES_POOL_SIZE,
            max_overflow=self.settings.POSTGRES_MAX_OVERFLOW,
            pool_pre_ping=True,
            echo=self.settings.DEBUG,
        )
        
        # Async engine (for API)
        self.async_engine = create_async_engine(
            self.settings.async_database_url,
            poolclass=QueuePool,
            pool_size=self.settings.POSTGRES_POOL_SIZE,
            max_overflow=self.settings.POSTGRES_MAX_OVERFLOW,
            pool_pre_ping=True,
            echo=self.settings.DEBUG,
        )
        
        # Session factories
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
            class_=Session
        )
        
        self.AsyncSessionLocal = async_sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.async_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Enable pgvector extension on connect
        self._setup_pgvector()
        
        logger.info("Database configuration initialized")
    
    def _setup_pgvector(self):
        """Set up pgvector extension"""
        @event.listens_for(self.engine, "connect")
        def setup_vector(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            try:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                dbapi_conn.commit()
            except Exception as e:
                logger.warning(f"Could not enable pgvector extension: {e}")
            finally:
                cursor.close()
    
    def get_session(self) -> Session:
        """Get synchronous database session"""
        return self.SessionLocal()
    
    async def get_async_session(self) -> AsyncSession:
        """Get asynchronous database session"""
        return self.AsyncSessionLocal()
    
    def create_tables(self):
        """Create all tables (sync)"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    async def create_tables_async(self):
        """Create all tables (async)"""
        try:
            async with self.async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully (async)")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    def drop_tables(self):
        """Drop all tables (sync) - USE WITH CAUTION"""
        if self.settings.ENVIRONMENT != "production":
            Base.metadata.drop_all(bind=self.engine)
            logger.warning("All database tables dropped")
        else:
            logger.error("Cannot drop tables in production environment")
    
    async def close(self):
        """Close database connections"""
        await self.async_engine.dispose()
        self.engine.dispose()
        logger.info("Database connections closed")


# Dependency for FastAPI
async def get_db():
    """FastAPI dependency for getting async database session"""
    db_config = DatabaseConfig()
    session = await db_config.get_async_session()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()