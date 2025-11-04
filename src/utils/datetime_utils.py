"""
Date and time utility functions.

This module provides common date/time operations used throughout the system,
including parsing, formatting, timezone handling, and temporal calculations.
"""

from typing import Optional, Union
from datetime import datetime, date, timedelta
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def parse_date(date_input: Union[str, datetime, date, None]) -> Optional[datetime]:
    """
    Parse various date input formats to datetime.
    
    Handles:
    - ISO format strings
    - datetime objects
    - date objects
    - None values
    
    Args:
        date_input: Input to parse (str, datetime, date, or None)
        
    Returns:
        Parsed datetime object, or None if input is None
        
    Raises:
        ValueError: If date string cannot be parsed
        
    Example:
        >>> parse_date("2024-01-15")
        datetime(2024, 1, 15, 0, 0)
        >>> parse_date("2024-01-15T10:30:00")
        datetime(2024, 1, 15, 10, 30, 0)
        >>> parse_date(datetime(2024, 1, 15))
        datetime(2024, 1, 15, 0, 0)
    """
    if date_input is None:
        return None
    
    if isinstance(date_input, datetime):
        return date_input
    
    if isinstance(date_input, date):
        return datetime.combine(date_input, datetime.min.time())
    
    if isinstance(date_input, str):
        try:
            # Try pandas parsing first (handles many formats)
            parsed = pd.to_datetime(date_input)
            if isinstance(parsed, pd.Timestamp):
                return parsed.to_pydatetime()
            return parsed
        except (ValueError, TypeError) as e:
            try:
                # Fallback to standard library
                return datetime.fromisoformat(date_input.replace('Z', '+00:00'))
            except ValueError:
                logger.error(f"Failed to parse date: {date_input}")
                raise ValueError(f"Invalid date format: {date_input}") from e
    
    raise ValueError(f"Unsupported date type: {type(date_input)}")


def to_utc(dt: datetime) -> datetime:
    """
    Convert datetime to UTC timezone.
    
    If datetime is naive (no timezone), assumes it's already UTC.
    
    Args:
        dt: Datetime to convert
        
    Returns:
        UTC datetime (timezone-aware or naive)
    """
    if dt is None:
        return None
    
    if dt.tzinfo is None:
        # Assume naive datetime is already UTC
        return dt
    
    return dt.astimezone(datetime.now().astimezone().utc).replace(tzinfo=None)


def format_iso(dt: Optional[datetime]) -> Optional[str]:
    """
    Format datetime as ISO 8601 string.
    
    Args:
        dt: Datetime to format
        
    Returns:
        ISO 8601 formatted string, or None if input is None
    """
    if dt is None:
        return None
    
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    
    return dt.isoformat()


def days_between(start_date: datetime, end_date: datetime) -> int:
    """
    Calculate number of days between two dates.
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        Number of days (can be negative if end_date < start_date)
    """
    if start_date is None or end_date is None:
        raise ValueError("Both dates must be provided")
    
    delta = end_date - start_date
    return delta.days


def months_between(start_date: datetime, end_date: datetime) -> float:
    """
    Calculate approximate number of months between two dates.
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        Number of months as float
    """
    if start_date is None or end_date is None:
        raise ValueError("Both dates must be provided")
    
    days = days_between(start_date, end_date)
    return days / 30.44  # Average days per month


def is_date_in_range(
    check_date: datetime,
    start_date: datetime,
    end_date: datetime,
    inclusive: bool = True
) -> bool:
    """
    Check if a date falls within a date range.
    
    Args:
        check_date: Date to check
        start_date: Range start date
        end_date: Range end date
        inclusive: If True, includes boundaries; if False, excludes them
        
    Returns:
        True if date is in range, False otherwise
    """
    if check_date is None or start_date is None or end_date is None:
        return False
    
    if inclusive:
        return start_date <= check_date <= end_date
    else:
        return start_date < check_date < end_date


def get_current_utc() -> datetime:
    """
    Get current UTC datetime (naive).
    
    Returns:
        Current UTC datetime without timezone info
    """
    return datetime.utcnow()


def add_days(dt: datetime, days: int) -> datetime:
    """
    Add days to a datetime.
    
    Args:
        dt: Base datetime
        days: Number of days to add (can be negative)
        
    Returns:
        New datetime with days added
    """
    if dt is None:
        raise ValueError("Datetime cannot be None")
    
    return dt + timedelta(days=days)


def add_months(dt: datetime, months: int) -> datetime:
    """
    Add months to a datetime (approximate).
    
    Uses average month length of 30.44 days.
    
    Args:
        dt: Base datetime
        months: Number of months to add (can be negative)
        
    Returns:
        New datetime with months added
    """
    if dt is None:
        raise ValueError("Datetime cannot be None")
    
    days = int(months * 30.44)
    return dt + timedelta(days=days)


def get_age_in_days(birth_date: datetime) -> Optional[int]:
    """
    Calculate age in days from birth date to now.
    
    Args:
        birth_date: Birth/reference date
        
    Returns:
        Age in days, or None if birth_date is None
    """
    if birth_date is None:
        return None
    
    return days_between(birth_date, get_current_utc())


def get_season(dt: datetime) -> str:
    """
    Get season name for a given date.
    
    Args:
        dt: Datetime to check
        
    Returns:
        Season name: 'SPRING', 'SUMMER', 'MONSOON', 'WINTER'
    """
    if dt is None:
        raise ValueError("Datetime cannot be None")
    
    month = dt.month
    
    # Indian seasons
    if month in [3, 4, 5]:  # March, April, May
        return 'SUMMER'
    elif month in [6, 7, 8, 9]:  # June, July, August, September
        return 'MONSOON'
    elif month in [10, 11]:  # October, November
        return 'SPRING'
    else:  # December, January, February
        return 'WINTER'


def is_weekend(dt: datetime) -> bool:
    """
    Check if date falls on weekend.
    
    Args:
        dt: Datetime to check
        
    Returns:
        True if weekend (Saturday or Sunday), False otherwise
    """
    if dt is None:
        raise ValueError("Datetime cannot be None")
    
    return dt.weekday() >= 5  # 5 = Saturday, 6 = Sunday

