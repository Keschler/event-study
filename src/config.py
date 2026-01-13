"""Configuration loader and validator for the event study pipeline."""

import json
from pathlib import Path
from typing import Any, Dict, List


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Load and validate configuration from JSON file.
    
    Args:
        config_path: Path to the configuration file (default: config.json)
        
    Returns:
        Dictionary containing validated configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If required keys are missing or invalid
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            "Please create a config.json file with required settings."
        )
    
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # Validate required top-level keys
    required_keys = ["scrape", "paths", "market", "event"]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(
            f"Missing required configuration sections: {', '.join(missing_keys)}"
        )
    
    # Validate scrape section
    _validate_section(config, "scrape", ["username", "max_posts", "cookies_path"])
    
    # Validate paths section
    _validate_section(config, "paths", ["raw_dir", "processed_dir", "outputs_dir"])
    
    # Validate market section
    _validate_section(config, "market", ["benchmark"])
    
    # Validate event section
    _validate_section(
        config, "event", 
        ["event_window", "estimation_window", "market_tz"]
    )
    
    # Validate event_window and estimation_window are lists with 2 elements
    if not isinstance(config["event"]["event_window"], list) or \
       len(config["event"]["event_window"]) != 2:
        raise ValueError(
            "event.event_window must be a list with 2 elements [start, end]"
        )
    
    if not isinstance(config["event"]["estimation_window"], list) or \
       len(config["event"]["estimation_window"]) != 2:
        raise ValueError(
            "event.estimation_window must be a list with 2 elements [start, end]"
        )
    
    return config


def _validate_section(config: Dict[str, Any], section: str, 
                     required_fields: List[str]) -> None:
    """
    Validate that a configuration section has all required fields.
    
    Args:
        config: Configuration dictionary
        section: Section name to validate
        required_fields: List of required field names
        
    Raises:
        ValueError: If required fields are missing
    """
    if section not in config:
        raise ValueError(f"Missing configuration section: {section}")
    
    section_data = config[section]
    missing_fields = [
        field for field in required_fields 
        if field not in section_data
    ]
    
    if missing_fields:
        raise ValueError(
            f"Missing required fields in [{section}]: {', '.join(missing_fields)}"
        )


def get_market_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract market-related configuration."""
    return config["market"]


def get_event_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract event study-related configuration."""
    return config["event"]


def get_scrape_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract scraping-related configuration."""
    return config["scrape"]


def get_paths_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract paths-related configuration."""
    return config["paths"]
