import redis.asyncio as redis
from typing import Optional
import logging

from .settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RedisConfig:
    """Redis configuration and connection management"""
    
    def __init__(self):
        self.settings = settings
        self.client: Optional[redis.Redis] = None
        self.pool: Optional[redis.ConnectionPool] = None
    
    async def connect(self) -> redis.Redis:
        """Create and return Redis client"""
        try:
            # Create connection pool
            self.pool = redis.ConnectionPool(
                host=self.settings.REDIS_HOST,
                port=self.settings.REDIS_PORT,
                db=self.settings.REDIS_DB,
                password=self.settings.REDIS_PASSWORD,
                max_connections=self.settings.REDIS_MAX_CONNECTIONS,
                decode_responses=self.settings.REDIS_DECODE_RESPONSES,
                socket_timeout=self.settings.REDIS_SOCKET_TIMEOUT,
                socket_connect_timeout=self.settings.REDIS_SOCKET_CONNECT_TIMEOUT,
            )
            
            # Create client
            self.client = redis.Redis(connection_pool=self.pool)
            
            # Test connection
            await self.client.ping()
            
            logger.info(
                f"Connected to Redis at {self.settings.REDIS_HOST}:"
                f"{self.settings.REDIS_PORT}"
            )
            
            return self.client
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def disconnect(self):
        """Close Redis connection"""
        try:
            if self.client:
                await self.client.close()
            if self.pool:
                await self.pool.disconnect()
            logger.info("Disconnected from Redis")
        except Exception as e:
            logger.error(f"Error disconnecting from Redis: {e}")
    
    async def health_check(self) -> bool:
        """Check if Redis is healthy"""
        try:
            if not self.client:
                return False
            await self.client.ping()
            return True
        except:
            return False
    
    async def get_info(self) -> dict:
        """Get Redis server information"""
        if not self.client:
            return {}
        try:
            info = await self.client.info()
            return {
                "version": info.get("redis_version"),
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "uptime_days": info.get("uptime_in_days"),
            }
        except Exception as e:
            logger.error(f"Error getting Redis info: {e}")
            return {}
    
    async def flush_db(self, db: Optional[int] = None):
        """Flush Redis database - USE WITH CAUTION"""
        if self.settings.ENVIRONMENT != "production":
            if not self.client:
                return
            await self.client.flushdb()
            logger.warning(f"Redis database {db or self.settings.REDIS_DB} flushed")
        else:
            logger.error("Cannot flush Redis in production environment")


# Singleton instance
_redis_config: Optional[RedisConfig] = None


async def get_redis() -> redis.Redis:
    """Get Redis client instance"""
    global _redis_config
    if _redis_config is None:
        _redis_config = RedisConfig()
        await _redis_config.connect()
    return _redis_config.client


async def close_redis():
    """Close Redis connection"""
    global _redis_config
    if _redis_config:
        await _redis_config.disconnect()
        _redis_config = None