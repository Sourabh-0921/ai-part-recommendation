"""
JSON serialization and deserialization utilities.

This module provides safe JSON operations with proper error handling,
datetime serialization, and custom encoding support.
"""

import json
from typing import Any, Optional, Dict, Union
from datetime import datetime, date
import logging

logger = logging.getLogger(__name__)


class DateTimeEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles datetime and date objects.
    
    Converts:
    - datetime -> ISO format string
    - date -> ISO format string
    - Other types -> falls back to default encoding
    """
    
    def default(self, obj: Any) -> Any:
        """Encode custom objects to JSON-serializable format."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, date):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            # For custom objects with __dict__
            return obj.__dict__
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
            # For iterables (but not strings)
            try:
                return list(obj)
            except TypeError:
                pass
        
        # Let base class handle the rest
        return super().default(obj)


def safe_dumps(
    obj: Any,
    indent: Optional[int] = None,
    ensure_ascii: bool = False,
    default: Any = None,
    **kwargs
) -> str:
    """
    Safely serialize Python object to JSON string.
    
    Handles datetime/date objects, custom objects, and provides
    proper error handling.
    
    Args:
        obj: Object to serialize
        indent: Optional indentation for pretty printing
        ensure_ascii: If True, escape non-ASCII characters
        default: Optional callable for custom serialization
        **kwargs: Additional arguments to pass to json.dumps
        
    Returns:
        JSON string representation
        
    Raises:
        TypeError: If object cannot be serialized
        
    Example:
        >>> safe_dumps({'date': datetime(2024, 1, 15)})
        '{"date": "2024-01-15T00:00:00"}'
        >>> safe_dumps({'key': 'value'}, indent=2)
        '{\\n  "key": "value"\\n}'
    """
    try:
        if default is None:
            default = DateTimeEncoder().default
        
        return json.dumps(
            obj,
            indent=indent,
            ensure_ascii=ensure_ascii,
            default=default,
            **kwargs
        )
    except (TypeError, ValueError) as e:
        logger.error(f"Failed to serialize object to JSON: {e}")
        raise TypeError(f"Object not JSON serializable: {e}") from e


def safe_loads(
    json_str: Union[str, bytes],
    default: Any = None,
    **kwargs
) -> Any:
    """
    Safely deserialize JSON string to Python object.
    
    Provides proper error handling and optional default value.
    
    Args:
        json_str: JSON string or bytes to parse
        default: Optional default value if parsing fails
        **kwargs: Additional arguments to pass to json.loads
        
    Returns:
        Deserialized Python object, or default if parsing fails and default provided
        
    Raises:
        json.JSONDecodeError: If JSON is invalid and no default provided
        
    Example:
        >>> safe_loads('{"key": "value"}')
        {'key': 'value'}
        >>> safe_loads('invalid json', default={})
        {}
    """
    if json_str is None:
        return default
    
    if isinstance(json_str, bytes):
        json_str = json_str.decode('utf-8')
    
    try:
        return json.loads(json_str, **kwargs)
    except json.JSONDecodeError as e:
        if default is not None:
            logger.warning(f"Failed to parse JSON, using default: {e}")
            return default
        logger.error(f"Invalid JSON string: {e}")
        raise


def safe_load_file(file_path: str, default: Any = None, **kwargs) -> Any:
    """
    Safely load JSON from file.
    
    Args:
        file_path: Path to JSON file
        default: Optional default value if file read/parse fails
        **kwargs: Additional arguments to pass to json.loads
        
    Returns:
        Deserialized Python object, or default if loading fails and default provided
        
    Raises:
        FileNotFoundError: If file doesn't exist and no default provided
        json.JSONDecodeError: If JSON is invalid and no default provided
        
    Example:
        >>> safe_load_file('config.json')
        {'key': 'value'}
        >>> safe_load_file('missing.json', default={})
        {}
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f, **kwargs)
    except FileNotFoundError:
        if default is not None:
            logger.warning(f"File not found, using default: {file_path}")
            return default
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        if default is not None:
            logger.warning(f"Invalid JSON in file, using default: {file_path}")
            return default
        logger.error(f"Invalid JSON in file {file_path}: {e}")
        raise
    except Exception as e:
        if default is not None:
            logger.warning(f"Error reading file, using default: {file_path}")
            return default
        logger.error(f"Error reading file {file_path}: {e}")
        raise


def safe_dump_file(obj: Any, file_path: str, indent: Optional[int] = None, **kwargs) -> None:
    """
    Safely write Python object to JSON file.
    
    Args:
        obj: Object to serialize
        file_path: Path to output file
        indent: Optional indentation for pretty printing
        **kwargs: Additional arguments to pass to json.dumps
        
    Raises:
        IOError: If file cannot be written
        TypeError: If object cannot be serialized
        
    Example:
        >>> safe_dump_file({'key': 'value'}, 'output.json', indent=2)
    """
    try:
        json_str = safe_dumps(obj, indent=indent, **kwargs)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
        logger.debug(f"Successfully wrote JSON to {file_path}")
    except (IOError, OSError) as e:
        logger.error(f"Failed to write JSON file {file_path}: {e}")
        raise
    except TypeError as e:
        logger.error(f"Failed to serialize object to JSON: {e}")
        raise


def is_valid_json(json_str: Union[str, bytes]) -> bool:
    """
    Check if string is valid JSON.
    
    Args:
        json_str: String or bytes to check
        
    Returns:
        True if valid JSON, False otherwise
        
    Example:
        >>> is_valid_json('{"key": "value"}')
        True
        >>> is_valid_json('invalid json')
        False
    """
    try:
        safe_loads(json_str)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def parse_datetime_from_json(data: Dict[str, Any], key: str) -> Optional[datetime]:
    """
    Parse datetime from JSON data (handles ISO format strings).
    
    Args:
        data: Dictionary containing the datetime
        key: Key to extract datetime from
        
    Returns:
        Parsed datetime object, or None if key not found or value is None
        
    Raises:
        ValueError: If value cannot be parsed as datetime
        
    Example:
        >>> parse_datetime_from_json({'date': '2024-01-15T10:30:00'}, 'date')
        datetime(2024, 1, 15, 10, 30, 0)
    """
    if key not in data:
        return None
    
    value = data[key]
    if value is None:
        return None
    
    if isinstance(value, datetime):
        return value
    
    if isinstance(value, str):
        try:
            # Import here to avoid circular import
            from .datetime_utils import parse_date
            return parse_date(value)
        except ValueError as e:
            logger.error(f"Failed to parse datetime from JSON key '{key}': {e}")
            raise ValueError(f"Invalid datetime format for key '{key}': {value}") from e
    
    raise ValueError(f"Expected string or datetime for key '{key}', got {type(value)}")

