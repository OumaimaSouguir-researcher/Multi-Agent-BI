import os
from typing import Optional
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class Settings(BaseSettings):
    """
    Application-wide settings loaded from environment variables
    """
    
    # Application
    APP_NAME: str = "Multi-Agent Business Intelligence"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    
    # API Configuration
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    API_PORT: int = Field(default=8000, env="API_PORT")
    API_RELOAD: bool = Field(default=True, env="API_RELOAD")
    API_WORKERS: int = Field(default=1, env="API_WORKERS")
    
    # CORS
    CORS_ORIGINS: list = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        env="CORS_ORIGINS"
    )
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list = ["*"]
    CORS_ALLOW_HEADERS: list = ["*"]
    
    # Database (PostgreSQL)
    POSTGRES_HOST: str = Field(default="localhost", env="POSTGRES_HOST")
    POSTGRES_PORT: int = Field(default=5432, env="POSTGRES_PORT")
    POSTGRES_USER: str = Field(default="postgres", env="POSTGRES_USER")
    POSTGRES_PASSWORD: str = Field(default="postgres", env="POSTGRES_PASSWORD")
    POSTGRES_DB: str = Field(default="agent_system", env="POSTGRES_DB")
    POSTGRES_POOL_SIZE: int = Field(default=10, env="POSTGRES_POOL_SIZE")
    POSTGRES_MAX_OVERFLOW: int = Field(default=20, env="POSTGRES_MAX_OVERFLOW")
    
    @property
    def database_url(self) -> str:
        """Generate PostgreSQL connection URL"""
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )
    
    @property
    def async_database_url(self) -> str:
        """Generate async PostgreSQL connection URL"""
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )
    
    # Redis
    REDIS_HOST: str = Field(default="localhost", env="REDIS_HOST")
    REDIS_PORT: int = Field(default=6379, env="REDIS_PORT")
    REDIS_DB: int = Field(default=0, env="REDIS_DB")
    REDIS_PASSWORD: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    REDIS_MAX_CONNECTIONS: int = Field(default=50, env="REDIS_MAX_CONNECTIONS")
    REDIS_DECODE_RESPONSES: bool = True
    REDIS_SOCKET_TIMEOUT: int = 5
    REDIS_SOCKET_CONNECT_TIMEOUT: int = 5
    
    # Ollama (LLM)
    OLLAMA_HOST: str = Field(default="http://localhost:11434", env="OLLAMA_HOST")
    OLLAMA_MODEL: str = Field(default="llama3.2:3b", env="OLLAMA_MODEL")
    OLLAMA_EMBEDDING_MODEL: str = Field(default="nomic-embed-text", env="OLLAMA_EMBEDDING_MODEL")
    OLLAMA_TEMPERATURE: float = Field(default=0.7, env="OLLAMA_TEMPERATURE")
    OLLAMA_TOP_P: float = Field(default=0.9, env="OLLAMA_TOP_P")
    OLLAMA_MAX_TOKENS: int = Field(default=2000, env="OLLAMA_MAX_TOKENS")
    OLLAMA_TIMEOUT: int = Field(default=120, env="OLLAMA_TIMEOUT")
    
    # Vector Store (pgvector)
    VECTOR_DIMENSION: int = Field(default=768, env="VECTOR_DIMENSION")
    VECTOR_SIMILARITY_THRESHOLD: float = Field(default=0.7, env="VECTOR_SIMILARITY_THRESHOLD")
    VECTOR_TOP_K: int = Field(default=5, env="VECTOR_TOP_K")
    
    # Memory Configuration
    SHORT_TERM_MEMORY_TTL: int = Field(default=3600, env="SHORT_TERM_MEMORY_TTL")  # 1 hour
    LONG_TERM_MEMORY_RETENTION: int = Field(default=90, env="LONG_TERM_MEMORY_RETENTION")  # 90 days
    EPISODIC_MEMORY_MAX_SIZE: int = Field(default=1000, env="EPISODIC_MEMORY_MAX_SIZE")
    
    # Message Queue
    MESSAGE_QUEUE_MAX_SIZE: int = Field(default=10000, env="MESSAGE_QUEUE_MAX_SIZE")
    MESSAGE_TTL: int = Field(default=3600, env="MESSAGE_TTL")
    MESSAGE_BATCH_SIZE: int = Field(default=100, env="MESSAGE_BATCH_SIZE")
    
    # Agent Configuration
    MAX_AGENT_RETRIES: int = Field(default=3, env="MAX_AGENT_RETRIES")
    AGENT_TIMEOUT: int = Field(default=300, env="AGENT_TIMEOUT")  # 5 minutes
    AGENT_HEARTBEAT_INTERVAL: int = Field(default=30, env="AGENT_HEARTBEAT_INTERVAL")
    
    # Task Configuration
    MAX_TASK_DURATION: int = Field(default=3600, env="MAX_TASK_DURATION")  # 1 hour
    TASK_CLEANUP_INTERVAL: int = Field(default=300, env="TASK_CLEANUP_INTERVAL")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    LOG_FILE: Optional[str] = Field(default="logs/agent_system.log", env="LOG_FILE")
    LOG_FILE_MAX_BYTES: int = Field(default=10485760, env="LOG_FILE_MAX_BYTES")  # 10MB
    LOG_FILE_BACKUP_COUNT: int = Field(default=5, env="LOG_FILE_BACKUP_COUNT")
    
    # Security
    SECRET_KEY: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    ALGORITHM: str = Field(default="HS256", env="ALGORITHM")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    RATE_LIMIT_REQUESTS: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    RATE_LIMIT_PERIOD: int = Field(default=60, env="RATE_LIMIT_PERIOD")  # seconds
    
    # Monitoring
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    METRICS_PORT: int = Field(default=9090, env="METRICS_PORT")
    
    # File Storage
    UPLOAD_DIR: str = Field(default="uploads", env="UPLOAD_DIR")
    MAX_UPLOAD_SIZE: int = Field(default=10485760, env="MAX_UPLOAD_SIZE")  # 10MB
    ALLOWED_EXTENSIONS: list = Field(
        default=["txt", "pdf", "csv", "json", "xlsx"],
        env="ALLOWED_EXTENSIONS"
    )
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()
    
    @validator("ENVIRONMENT")
    def validate_environment(cls, v):
        valid_envs = ["development", "staging", "production"]
        if v.lower() not in valid_envs:
            raise ValueError(f"ENVIRONMENT must be one of {valid_envs}")
        return v.lower()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance
    Use this function to access settings throughout the application
    """
    return Settings()