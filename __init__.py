"""
Projet d'application Python
============================

Ce package contient l'application principale et ses modules utilitaires.

Author: SouguirOumaima
Version: 1.0.0
License: MIT
"""

import sys
import os
from pathlib import Path

# Version du projet
__version__ = "1.0.0"
__author__ = "Votre Nom"
__license__ = "MIT"

# Ajout du répertoire parent au PYTHONPATH pour faciliter les imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import des modules principaux
try:
    from .utils import (
        setup_logger,
        get_logger,
        AppException,
        ValidationError,
        DatabaseError,
        APIError,
        MetricsCollector,
        track_time,
        count_calls,
        validate_email,
        validate_phone,
        validate_required_fields,
    )
except ImportError:
    # Si les imports relatifs échouent, on continue sans erreur
    pass

# Configuration par défaut
DEFAULT_CONFIG = {
    "app_name": "MyApp",
    "debug": False,
    "log_level": "INFO",
    "log_dir": "logs",
    "data_dir": "data",
    "temp_dir": "temp",
}

# Initialisation du logger principal
try:
    logger = setup_logger(
        name="app",
        level=os.getenv("LOG_LEVEL", "INFO"),
        log_file="app.log",
        console_output=True,
        colored_output=True
    )
    logger.info(f"Application initialisée - Version {__version__}")
except Exception as e:
    print(f"Erreur lors de l'initialisation du logger: {e}")

# Création des répertoires nécessaires
def _create_directories():
    """Crée les répertoires nécessaires pour l'application."""
    directories = [
        DEFAULT_CONFIG["log_dir"],
        DEFAULT_CONFIG["data_dir"],
        DEFAULT_CONFIG["temp_dir"],
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

# Initialisation au chargement du module
try:
    _create_directories()
except Exception as e:
    print(f"Erreur lors de la création des répertoires: {e}")

# Export des éléments principaux
__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "DEFAULT_CONFIG",
    "PROJECT_ROOT",
    # Fonctions et classes utilitaires (si importées avec succès)
    "setup_logger",
    "get_logger",
    "AppException",
    "ValidationError",
    "DatabaseError",
    "APIError",
    "MetricsCollector",
    "track_time",
    "count_calls",
    "validate_email",
    "validate_phone",
    "validate_required_fields",
]


def get_version():
    """Retourne la version de l'application."""
    return __version__


def get_project_info():
    """Retourne les informations sur le projet."""
    return {
        "name": DEFAULT_CONFIG["app_name"],
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "project_root": str(PROJECT_ROOT),
    }


# Message de bienvenue (optionnel, peut être commenté en production)
if __name__ != "__main__":
    try:
        logger.debug(f"Module {__name__} chargé avec succès")
    except:
        pass