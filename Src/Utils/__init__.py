"""
Module utilitaire pour l'application.
Fournit des fonctionnalités communes : logging, métriques, validation et gestion des exceptions.
"""

from .logger import setup_logger, get_logger
from .exceptions import (
    AppException,
    ValidationError,
    DatabaseError,
    APIError,
    ConfigurationError,
    NotFoundError
)
from .metrics import MetricsCollector, track_time, count_calls
from .validators import (
    validate_email,
    validate_phone,
    validate_url,
    validate_date,
    validate_required_fields,
    sanitize_input
)

__version__ = "1.0.0"
__all__ = [
    # Logger
    "setup_logger",
    "get_logger",
    # Exceptions
    "AppException",
    "ValidationError",
    "DatabaseError",
    "APIError",
    "ConfigurationError",
    "NotFoundError",
    # Metrics
    "MetricsCollector",
    "track_time",
    "count_calls",
    # Validators
    "validate_email",
    "validate_phone",
    "validate_url",
    "validate_date",
    "validate_required_fields",
    "sanitize_input",
]