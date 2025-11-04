"""
Data validation utility functions.

This module provides common validation functions for checking data types,
ranges, formats, and business rules.
"""

from typing import Any, Optional, List, Dict, Union, Callable
import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def is_valid_vehicle_id(vehicle_id: Any) -> bool:
    """
    Validate vehicle ID format.
    
    Expected format: alphanumeric string, 6-20 characters
    
    Args:
        vehicle_id: Value to validate
        
    Returns:
        True if valid vehicle ID format, False otherwise
        
    Example:
        >>> is_valid_vehicle_id("VH123456")
        True
        >>> is_valid_vehicle_id("")
        False
        >>> is_valid_vehicle_id(None)
        False
    """
    if not isinstance(vehicle_id, str):
        return False
    
    if not vehicle_id:
        return False
    
    # Alphanumeric, 6-20 characters
    pattern = r'^[A-Za-z0-9]{6,20}$'
    return bool(re.match(pattern, vehicle_id))


def is_valid_part_code(part_code: Any) -> bool:
    """
    Validate part code format.
    
    Expected format: 2 letters followed by 3 digits (e.g., "BP001", "AF123")
    
    Args:
        part_code: Value to validate
        
    Returns:
        True if valid part code format, False otherwise
        
    Example:
        >>> is_valid_part_code("BP001")
        True
        >>> is_valid_part_code("BP1")
        False
        >>> is_valid_part_code("123")
        False
    """
    if not isinstance(part_code, str):
        return False
    
    if not part_code:
        return False
    
    # 2 letters + 3 digits
    pattern = r'^[A-Z]{2}\d{3}$'
    return bool(re.match(pattern, part_code.upper()))


def is_valid_odometer(odometer: Any, min_value: float = 0.0, max_value: float = 9999999.0) -> bool:
    """
    Validate odometer reading.
    
    Args:
        odometer: Value to validate
        min_value: Minimum valid value (default: 0.0)
        max_value: Maximum valid value (default: 9999999.0)
        
    Returns:
        True if valid odometer reading, False otherwise
        
    Example:
        >>> is_valid_odometer(50000.5)
        True
        >>> is_valid_odometer(-100)
        False
        >>> is_valid_odometer("50000")
        False
    """
    if not isinstance(odometer, (int, float)):
        return False
    
    if isinstance(odometer, float) and odometer.isnan():
        return False
    
    return min_value <= odometer <= max_value


def is_valid_confidence_score(score: Any, min_value: float = 0.0, max_value: float = 100.0) -> bool:
    """
    Validate confidence score.
    
    Args:
        score: Value to validate
        min_value: Minimum valid value (default: 0.0)
        max_value: Maximum valid value (default: 100.0)
        
    Returns:
        True if valid confidence score, False otherwise
        
    Example:
        >>> is_valid_confidence_score(85.5)
        True
        >>> is_valid_confidence_score(150)
        False
        >>> is_valid_confidence_score(-10)
        False
    """
    if not isinstance(score, (int, float)):
        return False
    
    if isinstance(score, float) and score.isnan():
        return False
    
    return min_value <= score <= max_value


def is_valid_email(email: Any) -> bool:
    """
    Validate email address format.
    
    Args:
        email: Value to validate
        
    Returns:
        True if valid email format, False otherwise
        
    Example:
        >>> is_valid_email("user@example.com")
        True
        >>> is_valid_email("invalid.email")
        False
    """
    if not isinstance(email, str):
        return False
    
    if not email:
        return False
    
    # Simple email regex (not exhaustive but covers most cases)
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def is_valid_phone(phone: Any) -> bool:
    """
    Validate phone number format (Indian format).
    
    Accepts: 10 digits, optionally with +91 prefix
    
    Args:
        phone: Value to validate
        
    Returns:
        True if valid phone format, False otherwise
        
    Example:
        >>> is_valid_phone("9876543210")
        True
        >>> is_valid_phone("+919876543210")
        True
        >>> is_valid_phone("123")
        False
    """
    if not isinstance(phone, str):
        return False
    
    if not phone:
        return False
    
    # Remove spaces and common separators
    cleaned = re.sub(r'[\s\-\(\)]', '', phone)
    
    # Check for +91 prefix or direct 10-digit number
    pattern = r'^(\+91)?[6-9]\d{9}$'
    return bool(re.match(pattern, cleaned))


def validate_not_none(value: Any, name: str = "value") -> None:
    """
    Validate that value is not None.
    
    Args:
        value: Value to check
        name: Name of the value (for error message)
        
    Raises:
        ValueError: If value is None
        
    Example:
        >>> validate_not_none("test", "name")
        >>> validate_not_none(None, "name")
        ValueError: name cannot be None
    """
    if value is None:
        raise ValueError(f"{name} cannot be None")


def validate_not_empty(value: Any, name: str = "value") -> None:
    """
    Validate that value is not None and not empty.
    
    Args:
        value: Value to check (str, list, dict, etc.)
        name: Name of the value (for error message)
        
    Raises:
        ValueError: If value is None or empty
        
    Example:
        >>> validate_not_empty("test", "name")
        >>> validate_not_empty("", "name")
        ValueError: name cannot be empty
        >>> validate_not_empty([], "list")
        ValueError: list cannot be empty
    """
    if value is None:
        raise ValueError(f"{name} cannot be None")
    
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{name} cannot be empty")
    
    if isinstance(value, (list, dict, tuple, set)) and len(value) == 0:
        raise ValueError(f"{name} cannot be empty")


def validate_in_range(
    value: Union[int, float],
    min_value: Union[int, float],
    max_value: Union[int, float],
    name: str = "value"
) -> None:
    """
    Validate that numeric value is within range.
    
    Args:
        value: Value to check
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        name: Name of the value (for error message)
        
    Raises:
        ValueError: If value is outside range
        
    Example:
        >>> validate_in_range(50, 0, 100, "score")
        >>> validate_in_range(150, 0, 100, "score")
        ValueError: score must be between 0 and 100, got 150
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    
    if value < min_value or value > max_value:
        raise ValueError(
            f"{name} must be between {min_value} and {max_value}, got {value}"
        )


def validate_one_of(value: Any, allowed_values: List[Any], name: str = "value") -> None:
    """
    Validate that value is one of allowed values.
    
    Args:
        value: Value to check
        allowed_values: List of allowed values
        name: Name of the value (for error message)
        
    Raises:
        ValueError: If value is not in allowed values
        
    Example:
        >>> validate_one_of("red", ["red", "green", "blue"], "color")
        >>> validate_one_of("yellow", ["red", "green", "blue"], "color")
        ValueError: color must be one of ['red', 'green', 'blue'], got 'yellow'
    """
    if value not in allowed_values:
        raise ValueError(
            f"{name} must be one of {allowed_values}, got {value}"
        )


def validate_type(value: Any, expected_type: type, name: str = "value") -> None:
    """
    Validate that value is of expected type.
    
    Args:
        value: Value to check
        expected_type: Expected type
        name: Name of the value (for error message)
        
    Raises:
        TypeError: If value is not of expected type
        
    Example:
        >>> validate_type("test", str, "name")
        >>> validate_type(123, str, "name")
        TypeError: name must be of type <class 'str'>, got <class 'int'>
    """
    if not isinstance(value, expected_type):
        raise TypeError(
            f"{name} must be of type {expected_type}, got {type(value)}"
        )


def validate_list_of_type(
    values: List[Any],
    item_type: type,
    name: str = "list"
) -> None:
    """
    Validate that all items in list are of expected type.
    
    Args:
        values: List to validate
        item_type: Expected type for each item
        name: Name of the list (for error message)
        
    Raises:
        TypeError: If list contains items of wrong type
        
    Example:
        >>> validate_list_of_type([1, 2, 3], int, "numbers")
        >>> validate_list_of_type([1, "2", 3], int, "numbers")
        TypeError: All items in numbers must be of type <class 'int'>
    """
    validate_type(values, list, name)
    
    for i, item in enumerate(values):
        if not isinstance(item, item_type):
            raise TypeError(
                f"Item at index {i} in {name} must be of type {item_type}, "
                f"got {type(item)}"
            )


def sanitize_string(value: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize string by trimming whitespace and optionally truncating.
    
    Args:
        value: String to sanitize
        max_length: Optional maximum length (truncates if longer)
        
    Returns:
        Sanitized string
        
    Example:
        >>> sanitize_string("  test  ")
        'test'
        >>> sanitize_string("  test  ", max_length=2)
        'te'
    """
    if not isinstance(value, str):
        return str(value).strip()[:max_length] if max_length else str(value).strip()
    
    sanitized = value.strip()
    
    if max_length is not None and len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
        logger.debug(f"String truncated to {max_length} characters")
    
    return sanitized


def normalize_part_code(part_code: str) -> str:
    """
    Normalize part code to standard format (uppercase, no spaces).
    
    Args:
        part_code: Part code to normalize
        
    Returns:
        Normalized part code
        
    Example:
        >>> normalize_part_code("bp001")
        'BP001'
        >>> normalize_part_code("BP 001")
        'BP001'
    """
    if not isinstance(part_code, str):
        raise TypeError("part_code must be a string")
    
    # Remove spaces and convert to uppercase
    normalized = part_code.replace(" ", "").replace("-", "").upper()
    
    return normalized

