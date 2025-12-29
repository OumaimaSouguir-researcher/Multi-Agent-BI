"""
Fonctions de validation et de sanitisation des données.
"""

import re
from typing import Any, Dict, List, Optional
from datetime import datetime
from urllib.parse import urlparse
from .exceptions import ValidationError


def validate_email(email: str, required: bool = True) -> bool:
    """
    Valide une adresse email.
    
    Args:
        email: Adresse email à valider
        required: Si True, lève une exception si l'email est vide
    
    Returns:
        True si valide
    
    Raises:
        ValidationError: Si l'email n'est pas valide
    """
    if not email or not email.strip():
        if required:
            raise ValidationError("L'adresse email est requise", field="email")
        return True
    
    email = email.strip()
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not re.match(pattern, email):
        raise ValidationError(
            f"L'adresse email '{email}' n'est pas valide",
            field="email"
        )
    
    return True


def validate_phone(phone: str, required: bool = True, country_code: str = "TN") -> bool:
    """
    Valide un numéro de téléphone.
    
    Args:
        phone: Numéro de téléphone à valider
        required: Si True, lève une exception si le numéro est vide
        country_code: Code pays pour validation spécifique
    
    Returns:
        True si valide
    
    Raises:
        ValidationError: Si le numéro n'est pas valide
    """
    if not phone or not phone.strip():
        if required:
            raise ValidationError("Le numéro de téléphone est requis", field="phone")
        return True
    
    # Nettoie le numéro
    cleaned = re.sub(r'[\s\-\(\)]', '', phone)
    
    # Pattern général pour numéros internationaux
    if country_code == "TN":
        # Validation pour la Tunisie: +216 ou 00216 suivi de 8 chiffres
        pattern = r'^(\+216|00216|216)?[2459]\d{7}$'
    else:
        # Pattern général
        pattern = r'^(\+|00)?\d{8,15}$'
    
    if not re.match(pattern, cleaned):
        raise ValidationError(
            f"Le numéro de téléphone '{phone}' n'est pas valide",
            field="phone"
        )
    
    return True


def validate_url(url: str, required: bool = True, schemes: List[str] = None) -> bool:
    """
    Valide une URL.
    
    Args:
        url: URL à valider
        required: Si True, lève une exception si l'URL est vide
        schemes: Liste des schémas autorisés (http, https, etc.)
    
    Returns:
        True si valide
    
    Raises:
        ValidationError: Si l'URL n'est pas valide
    """
    if not url or not url.strip():
        if required:
            raise ValidationError("L'URL est requise", field="url")
        return True
    
    if schemes is None:
        schemes = ["http", "https"]
    
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            raise ValidationError(
                f"L'URL '{url}' n'est pas valide",
                field="url"
            )
        
        if result.scheme not in schemes:
            raise ValidationError(
                f"Le schéma '{result.scheme}' n'est pas autorisé. Schémas valides: {', '.join(schemes)}",
                field="url"
            )
        
        return True
    except Exception as e:
        raise ValidationError(
            f"L'URL '{url}' n'est pas valide: {str(e)}",
            field="url"
        )


def validate_date(
    date_str: str,
    date_format: str = "%Y-%m-%d",
    required: bool = True,
    min_date: Optional[datetime] = None,
    max_date: Optional[datetime] = None
) -> bool:
    """
    Valide une date.
    
    Args:
        date_str: Date sous forme de chaîne
        date_format: Format attendu de la date
        required: Si True, lève une exception si la date est vide
        min_date: Date minimale autorisée
        max_date: Date maximale autorisée
    
    Returns:
        True si valide
    
    Raises:
        ValidationError: Si la date n'est pas valide
    """
    if not date_str or not str(date_str).strip():
        if required:
            raise ValidationError("La date est requise", field="date")
        return True
    
    try:
        parsed_date = datetime.strptime(str(date_str).strip(), date_format)
        
        if min_date and parsed_date < min_date:
            raise ValidationError(
                f"La date doit être postérieure au {min_date.strftime(date_format)}",
                field="date"
            )
        
        if max_date and parsed_date > max_date:
            raise ValidationError(
                f"La date doit être antérieure au {max_date.strftime(date_format)}",
                field="date"
            )
        
        return True
    except ValueError:
        raise ValidationError(
            f"La date '{date_str}' n'est pas au format attendu ({date_format})",
            field="date"
        )


def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> bool:
    """
    Valide que tous les champs requis sont présents et non vides.
    
    Args:
        data: Dictionnaire de données à valider
        required_fields: Liste des champs requis
    
    Returns:
        True si tous les champs requis sont présents
    
    Raises:
        ValidationError: Si un champ requis est manquant ou vide
    """
    missing_fields = []
    empty_fields = []
    
    for field in required_fields:
        if field not in data:
            missing_fields.append(field)
        elif data[field] is None or (isinstance(data[field], str) and not data[field].strip()):
            empty_fields.append(field)
    
    if missing_fields:
        raise ValidationError(
            f"Champs manquants: {', '.join(missing_fields)}",
            details={"missing_fields": missing_fields}
        )
    
    if empty_fields:
        raise ValidationError(
            f"Champs vides: {', '.join(empty_fields)}",
            details={"empty_fields": empty_fields}
        )
    
    return True


def sanitize_input(text: str, max_length: Optional[int] = None, strip: bool = True) -> str:
    """
    Nettoie et sanitise une entrée texte.
    
    Args:
        text: Texte à nettoyer
        max_length: Longueur maximale (tronque si dépassée)
        strip: Si True, supprime les espaces en début et fin
    
    Returns:
        Texte nettoyé
    """
    if text is None:
        return ""
    
    text = str(text)
    
    if strip:
        text = text.strip()
    
    # Supprime les caractères de contrôle dangereux
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Tronque si nécessaire
    if max_length and len(text) > max_length:
        text = text[:max_length]
    
    return text


def validate_string_length(
    text: str,
    field_name: str,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None
) -> bool:
    """
    Valide la longueur d'une chaîne de caractères.
    
    Args:
        text: Texte à valider
        field_name: Nom du champ (pour les messages d'erreur)
        min_length: Longueur minimale
        max_length: Longueur maximale
    
    Returns:
        True si valide
    
    Raises:
        ValidationError: Si la longueur n'est pas valide
    """
    if text is None:
        text = ""
    
    length = len(text)
    
    if min_length is not None and length < min_length:
        raise ValidationError(
            f"Le champ '{field_name}' doit contenir au moins {min_length} caractères",
            field=field_name
        )
    
    if max_length is not None and length > max_length:
        raise ValidationError(
            f"Le champ '{field_name}' ne peut pas dépasser {max_length} caractères",
            field=field_name
        )
    
    return True


def validate_numeric_range(
    value: float,
    field_name: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None
) -> bool:
    """
    Valide qu'une valeur numérique est dans une plage donnée.
    
    Args:
        value: Valeur à valider
        field_name: Nom du champ (pour les messages d'erreur)
        min_value: Valeur minimale
        max_value: Valeur maximale
    
    Returns:
        True si valide
    
    Raises:
        ValidationError: Si la valeur n'est pas dans la plage
    """
    if min_value is not None and value < min_value:
        raise ValidationError(
            f"Le champ '{field_name}' doit être supérieur ou égal à {min_value}",
            field=field_name
        )
    
    if max_value is not None and value > max_value:
        raise ValidationError(
            f"Le champ '{field_name}' doit être inférieur ou égal à {max_value}",
            field=field_name
        )
    
    return True