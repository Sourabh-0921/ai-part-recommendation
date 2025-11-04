"""
Hashing utility functions.

This module provides consistent hashing functions for:
- A/B testing group assignment
- Cache key generation
- Consistent bucketing operations
"""

import hashlib
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)


def hash_string(text: str, algorithm: str = 'md5') -> str:
    """
    Hash a string using specified algorithm.
    
    Args:
        text: String to hash
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256', 'sha512')
        
    Returns:
        Hexadecimal hash string
        
    Raises:
        ValueError: If algorithm is not supported
        
    Example:
        >>> hash_string("test123")
        'cc03e747a6afbbcbf8be7668acfebee5'
    """
    if not text:
        raise ValueError("Text cannot be empty")
    
    algorithm = algorithm.lower()
    
    if algorithm == 'md5':
        hash_obj = hashlib.md5()
    elif algorithm == 'sha1':
        hash_obj = hashlib.sha1()
    elif algorithm == 'sha256':
        hash_obj = hashlib.sha256()
    elif algorithm == 'sha512':
        hash_obj = hashlib.sha512()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    hash_obj.update(text.encode('utf-8'))
    return hash_obj.hexdigest()


def hash_to_int(text: str, max_value: Optional[int] = None) -> int:
    """
    Hash a string to an integer.
    
    Useful for consistent bucketing (e.g., A/B testing).
    
    Args:
        text: String to hash
        max_value: Optional maximum value (for modulo operation)
        
    Returns:
        Integer hash value (or modulo if max_value provided)
        
    Example:
        >>> hash_to_int("vehicle_123")
        1234567890
        >>> hash_to_int("vehicle_123", max_value=100)
        90  # Consistent bucket assignment
    """
    if not text:
        raise ValueError("Text cannot be empty")
    
    # Use MD5 and take first 8 bytes as int
    hash_hex = hash_string(text, algorithm='md5')
    hash_int = int(hash_hex[:16], 16)  # First 16 hex chars = 8 bytes
    
    if max_value is not None:
        if max_value <= 0:
            raise ValueError("max_value must be positive")
        return hash_int % max_value
    
    return hash_int


def get_ab_test_group(
    identifier: str,
    split_ratio: float = 0.5,
    group_names: Optional[tuple] = None
) -> str:
    """
    Assign identifier to A/B test group consistently.
    
    Uses consistent hashing to ensure same identifier always gets
    same group assignment.
    
    Args:
        identifier: Unique identifier (e.g., vehicle_id, user_id)
        split_ratio: Ratio for group A (0.0 to 1.0). Default 0.5 = 50/50 split
        group_names: Optional tuple of group names (default: ('control', 'treatment'))
        
    Returns:
        Group name ('control' or 'treatment', or custom names)
        
    Raises:
        ValueError: If split_ratio is not between 0 and 1
        
    Example:
        >>> get_ab_test_group("vehicle_123", split_ratio=0.2)
        'control'  # 20% go to control, 80% to treatment
        >>> get_ab_test_group("vehicle_123", split_ratio=0.2)
        'control'  # Always same result for same identifier
    """
    if not identifier:
        raise ValueError("Identifier cannot be empty")
    
    if not 0.0 <= split_ratio <= 1.0:
        raise ValueError(f"split_ratio must be between 0 and 1, got {split_ratio}")
    
    if group_names is None:
        group_names = ('control', 'treatment')
    
    if len(group_names) != 2:
        raise ValueError("group_names must have exactly 2 elements")
    
    # Hash to integer in range 0-99
    hash_value = hash_to_int(identifier, max_value=100)
    
    # Assign based on split ratio
    threshold = int(split_ratio * 100)
    
    if hash_value < threshold:
        return group_names[0]  # control
    else:
        return group_names[1]  # treatment


def generate_cache_key(
    prefix: str,
    *args,
    separator: str = ':',
    **kwargs
) -> str:
    """
    Generate consistent cache key from components.
    
    Args:
        prefix: Key prefix (e.g., 'recommendation', 'vehicle')
        *args: Positional arguments to include in key
        separator: Separator between key components (default: ':')
        **kwargs: Keyword arguments to include in key (sorted for consistency)
        
    Returns:
        Cache key string
        
    Example:
        >>> generate_cache_key('recommendation', 'vehicle_123', odometer=50000)
        'recommendation:vehicle_123:odometer=50000'
        >>> generate_cache_key('vehicle', '123', 'features')
        'vehicle:123:features'
    """
    if not prefix:
        raise ValueError("Prefix cannot be empty")
    
    key_parts = [prefix]
    
    # Add positional arguments
    for arg in args:
        if arg is not None:
            key_parts.append(str(arg))
    
    # Add keyword arguments (sorted for consistency)
    if kwargs:
        sorted_kwargs = sorted(kwargs.items())
        for key, value in sorted_kwargs:
            if value is not None:
                key_parts.append(f"{key}={value}")
    
    return separator.join(key_parts)


def hash_dict(data: dict, algorithm: str = 'md5') -> str:
    """
    Hash a dictionary to a string (deterministic).
    
    Keys are sorted to ensure consistent hashing regardless of insertion order.
    
    Args:
        data: Dictionary to hash
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256', 'sha512')
        
    Returns:
        Hexadecimal hash string
        
    Example:
        >>> hash_dict({'a': 1, 'b': 2})
        '5d41402abc4b2a76b9719d911017c592'
        >>> hash_dict({'b': 2, 'a': 1})  # Same result despite different order
        '5d41402abc4b2a76b9719d911017c592'
    """
    if not data:
        return hash_string("{}", algorithm=algorithm)
    
    # Convert to sorted string representation
    sorted_items = sorted(data.items())
    dict_str = str(sorted_items)
    
    return hash_string(dict_str, algorithm=algorithm)

