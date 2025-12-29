"""
Exceptions personnalisées pour l'application.
"""


class AppException(Exception):
    """Classe de base pour toutes les exceptions de l'application."""
    
    def __init__(self, message: str, code: str = None, details: dict = None):
        self.message = message
        self.code = code or "APP_ERROR"
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> dict:
        """Convertit l'exception en dictionnaire."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "code": self.code,
            "details": self.details
        }


class ValidationError(AppException):
    """Erreur de validation des données."""
    
    def __init__(self, message: str, field: str = None, details: dict = None):
        code = "VALIDATION_ERROR"
        if field:
            details = details or {}
            details["field"] = field
        super().__init__(message, code, details)


class DatabaseError(AppException):
    """Erreur liée à la base de données."""
    
    def __init__(self, message: str, query: str = None, details: dict = None):
        code = "DATABASE_ERROR"
        if query:
            details = details or {}
            details["query"] = query
        super().__init__(message, code, details)


class APIError(AppException):
    """Erreur lors d'un appel API externe."""
    
    def __init__(self, message: str, status_code: int = None, endpoint: str = None, details: dict = None):
        code = "API_ERROR"
        details = details or {}
        if status_code:
            details["status_code"] = status_code
        if endpoint:
            details["endpoint"] = endpoint
        super().__init__(message, code, details)


class ConfigurationError(AppException):
    """Erreur de configuration de l'application."""
    
    def __init__(self, message: str, config_key: str = None, details: dict = None):
        code = "CONFIGURATION_ERROR"
        if config_key:
            details = details or {}
            details["config_key"] = config_key
        super().__init__(message, code, details)


class NotFoundError(AppException):
    """Erreur quand une ressource n'est pas trouvée."""
    
    def __init__(self, message: str, resource_type: str = None, resource_id: str = None, details: dict = None):
        code = "NOT_FOUND"
        details = details or {}
        if resource_type:
            details["resource_type"] = resource_type
        if resource_id:
            details["resource_id"] = resource_id
        super().__init__(message, code, details)


class AuthenticationError(AppException):
    """Erreur d'authentification."""
    
    def __init__(self, message: str = "Authentication failed", details: dict = None):
        code = "AUTHENTICATION_ERROR"
        super().__init__(message, code, details)


class AuthorizationError(AppException):
    """Erreur d'autorisation."""
    
    def __init__(self, message: str = "Access denied", required_permission: str = None, details: dict = None):
        code = "AUTHORIZATION_ERROR"
        if required_permission:
            details = details or {}
            details["required_permission"] = required_permission
        super().__init__(message, code, details)