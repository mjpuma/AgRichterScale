"""Utility functions for AgRichter framework."""

import os
import logging
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import json


def setup_logging(level: str = 'INFO', log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional log file path
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger('agrichter')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def validate_file_permissions(file_path: Union[str, Path], mode: str = 'r') -> bool:
    """
    Validate file permissions.
    
    Args:
        file_path: Path to file
        mode: Access mode ('r' for read, 'w' for write)
    
    Returns:
        True if file has required permissions
    """
    path = Path(file_path)
    
    if mode == 'r':
        return path.exists() and os.access(path, os.R_OK)
    elif mode == 'w':
        if path.exists():
            return os.access(path, os.W_OK)
        else:
            # Check if parent directory is writable
            return os.access(path.parent, os.W_OK)
    else:
        raise ValueError(f"Invalid mode '{mode}'. Use 'r' or 'w'.")


def validate_directory_structure(root_dir: Union[str, Path]) -> Dict[str, bool]:
    """
    Validate expected directory structure.
    
    Args:
        root_dir: Root directory path
    
    Returns:
        Dictionary with directory validation results
    """
    root_path = Path(root_dir)
    expected_dirs = ['inputs', 'ancillary', 'outputs']
    
    results = {}
    for dir_name in expected_dirs:
        dir_path = root_path / dir_name
        results[dir_name] = dir_path.exists() and dir_path.is_dir()
    
    return results


def create_directory_structure(root_dir: Union[str, Path]) -> None:
    """
    Create expected directory structure.
    
    Args:
        root_dir: Root directory path
    """
    root_path = Path(root_dir)
    directories = [
        'inputs',
        'ancillary', 
        'outputs',
        'outputs/media',
        'outputs/data'
    ]
    
    for dir_name in directories:
        dir_path = root_path / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent directory traversal and invalid characters.
    
    Args:
        filename: Input filename
    
    Returns:
        Sanitized filename
    """
    # Remove directory separators and other problematic characters
    invalid_chars = ['/', '\\', '..', ':', '*', '?', '"', '<', '>', '|']
    sanitized = filename
    
    for char in invalid_chars:
        sanitized = sanitized.replace(char, '_')
    
    # Remove leading/trailing whitespace and dots
    sanitized = sanitized.strip(' .')
    
    # Ensure filename is not empty
    if not sanitized:
        sanitized = 'unnamed_file'
    
    return sanitized


def validate_coordinate_ranges(lat: float, lon: float) -> bool:
    """
    Validate latitude and longitude ranges.
    
    Args:
        lat: Latitude value
        lon: Longitude value
    
    Returns:
        True if coordinates are valid
    """
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: File size in bytes
    
    Returns:
        Formatted file size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024.0 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def save_config_to_json(config_dict: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Save configuration dictionary to JSON file.
    
    Args:
        config_dict: Configuration dictionary
        output_path: Output file path
    """
    with open(output_path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)


def load_config_from_json(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Configuration file path
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return json.load(f)


def check_disk_space(path: Union[str, Path], required_gb: float = 1.0) -> bool:
    """
    Check if sufficient disk space is available.
    
    Args:
        path: Directory path to check
        required_gb: Required space in GB
    
    Returns:
        True if sufficient space is available
    """
    try:
        stat = os.statvfs(path)
        available_bytes = stat.f_bavail * stat.f_frsize
        required_bytes = required_gb * 1024**3
        return available_bytes >= required_bytes
    except (OSError, AttributeError):
        # Fallback for Windows or other systems
        import shutil
        available_bytes = shutil.disk_usage(path).free
        required_bytes = required_gb * 1024**3
        return available_bytes >= required_bytes


def find_closest_index(array: List[float], value: float) -> int:
    """
    Find index of closest value in array.
    
    Args:
        array: List of numeric values
        value: Target value
    
    Returns:
        Index of closest value
    """
    return min(range(len(array)), key=lambda i: abs(array[i] - value))


class ProgressReporter:
    """Simple progress reporter for long-running operations."""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        """
        Initialize progress reporter.
        
        Args:
            total_steps: Total number of steps
            description: Description of the operation
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.logger = logging.getLogger('agrichter')
    
    def update(self, step: int = 1, message: str = "") -> None:
        """
        Update progress.
        
        Args:
            step: Number of steps to advance
            message: Optional status message
        """
        self.current_step += step
        percentage = (self.current_step / self.total_steps) * 100
        
        status = f"{self.description}: {percentage:.1f}% ({self.current_step}/{self.total_steps})"
        if message:
            status += f" - {message}"
        
        self.logger.info(status)
    
    def complete(self, message: str = "Complete") -> None:
        """Mark operation as complete."""
        self.logger.info(f"{self.description}: {message}")


def validate_numeric_range(value: float, min_val: float, max_val: float, 
                          name: str = "value") -> bool:
    """
    Validate that a numeric value is within specified range.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name of the value for error messages
    
    Returns:
        True if value is valid
    
    Raises:
        ValueError: If value is outside valid range
    """
    if not min_val <= value <= max_val:
        raise ValueError(
            f"{name} must be between {min_val} and {max_val}, got {value}"
        )
    return True