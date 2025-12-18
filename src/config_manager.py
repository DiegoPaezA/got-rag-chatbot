"""Centralized Configuration Manager for the Game of Thrones RAG System.

This module provides a singleton ConfigManager class to load and access
configuration from cfg/config.json with proper defaults and type hints.

Usage:
    from src.config_manager import ConfigManager
    
    # Get root config
    config = ConfigManager.load("cfg/config.json")
    
    # Get nested values with defaults
    batch_size = ConfigManager.get("processing", "validator", "batch_size", default=10)
    
    # Get entire section
    paths = ConfigManager.get_section("paths")
"""

import json
import os
import logging
from typing import Any, Dict, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigManager:
    """Singleton configuration manager with hierarchical key access."""
    
    _instance: Optional['ConfigManager'] = None
    _config: Optional[Dict[str, Any]] = None
    _config_path: Optional[str] = None
    
    def __new__(cls) -> 'ConfigManager':
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def load(cls, config_path: str = "cfg/config.json") -> Dict[str, Any]:
        """Load configuration from JSON file.
        
        Args:
            config_path: Path to configuration file. Defaults to cfg/config.json.
            
        Returns:
            Dictionary containing the entire configuration.
            
        Raises:
            FileNotFoundError: If configuration file does not exist.
            json.JSONDecodeError: If configuration file is invalid JSON.
        """
        if cls._config is not None and cls._config_path == config_path:
            logger.debug(f"Using cached configuration from {config_path}")
            return cls._config
        
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                cls._config = json.load(f)
                cls._config_path = config_path
                logger.info(f"✅ Configuration loaded from {config_path}")
                return cls._config
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    @classmethod
    def get(cls, *keys: str, default: Any = None) -> Any:
        """Get a configuration value using hierarchical keys.
        
        Args:
            *keys: Variable-length argument list of keys to traverse the config tree.
            default: Default value if key path is not found.
            
        Returns:
            The configuration value at the specified key path, or default if not found.
            
        Examples:
            >>> ConfigManager.get("llm_settings", "model_name")
            "gemini-2.5-flash-preview-09-2025"
            
            >>> ConfigManager.get("processing", "validator", "batch_size", default=10)
            10
            
            >>> ConfigManager.get("nonexistent", "key", default="fallback")
            "fallback"
        """
        if cls._config is None:
            cls.load()
        
        value = cls._config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    logger.debug(f"Key path not found: {' -> '.join(keys)}. Using default: {default}")
                    return default
            else:
                logger.warning(f"Cannot traverse non-dict value at key: {key}")
                return default
        
        return value if value is not None else default
    
    @classmethod
    def get_section(cls, section: str) -> Dict[str, Any]:
        """Get an entire configuration section.
        
        Args:
            section: Name of the section to retrieve.
            
        Returns:
            Dictionary containing the entire section, or empty dict if not found.
            
        Examples:
            >>> paths = ConfigManager.get_section("paths")
            >>> processing = ConfigManager.get_section("llm_settings")
        """
        if cls._config is None:
            cls.load()
        
        return cls._config.get(section, {})
    
    @classmethod
    def get_all(cls) -> Dict[str, Any]:
        """Get the entire configuration.
        
        Returns:
            The complete configuration dictionary.
        """
        if cls._config is None:
            cls.load()
        
        return cls._config
    
    @classmethod
    def reload(cls) -> None:
        """Force reload configuration from disk."""
        config_path = cls._config_path or "cfg/config.json"
        cls._config = None
        cls._config_path = None
        cls.load(config_path)
        logger.info(f"Configuration reloaded from {config_path}")
    
    @classmethod
    def validate_required_keys(cls, required_keys: List[str]) -> bool:
        """Validate that all required configuration keys exist.
        
        Args:
            required_keys: List of dot-separated key paths to validate.
                Example: ["paths.raw_data", "database.neo4j.uri"]
        
        Returns:
            True if all keys exist, False otherwise.
        """
        if cls._config is None:
            cls.load()
        
        missing_keys = []
        for key_path in required_keys:
            keys = key_path.split('.')
            if cls.get(*keys) is None:
                missing_keys.append(key_path)
        
        if missing_keys:
            logger.error(f"Missing required configuration keys: {missing_keys}")
            return False
        
        logger.debug("All required configuration keys are present")
        return True
    
    @classmethod
    def get_paths(cls) -> Dict[str, Any]:
        """Convenience method to get all path configurations."""
        return cls.get_section("paths")
    
    @classmethod
    def get_llm_config(cls, component: str = "default") -> Dict[str, Any]:
        """Get LLM configuration for a specific component.
        
        Args:
            component: Component name (default, generator, augmenter, judge, validator).
        
        Returns:
            Dictionary with LLM configuration for the specified component.
        """
        if component == "default":
            return cls.get_section("llm_settings")
        
        # Try to get component-specific config
        component_config = cls.get("llm_settings", "components", component)
        if component_config:
            return component_config
        
        # Fall back to general llm_settings
        logger.warning(f"No component-specific config for '{component}', using default")
        return cls.get_section("llm_settings")
    
    @classmethod
    def get_embedding_config(cls) -> Dict[str, Any]:
        """Convenience method to get embedding configuration."""
        return cls.get_section("embedding_settings")
    
    @classmethod
    def get_database_config(cls) -> Dict[str, Any]:
        """Convenience method to get database configuration."""
        return cls.get_section("database")
    
    @classmethod
    def get_processing_config(cls, component: str = None) -> Dict[str, Any]:
        """Get processing configuration for a specific component.
        
        Args:
            component: Component name (validator, retriever, graph_search, inference).
                       If None, returns entire processing section.
        
        Returns:
            Dictionary with processing configuration.
        """
        if component is None:
            return cls.get_section("processing")
        
        component_config = cls.get("processing", component)
        return component_config if component_config else {}
    
    @classmethod
    def get_prompt(cls, prompt_type: str) -> str:
        """Get a prompt template by type.
        
        Args:
            prompt_type: Type of prompt (validator_system, validator_human, rag_response, etc.)
        
        Returns:
            The prompt template string.
        """
        return cls.get("prompts", prompt_type, default="")


# Helper functions for backward compatibility
def get_config(config_path: str = "cfg/config.json") -> Dict[str, Any]:
    """Load and return configuration dictionary.
    
    This is a convenience function that maintains backward compatibility
    with code expecting a simple function call.
    """
    return ConfigManager.load(config_path)


def get_config_value(*keys: str, default: Any = None, config_path: str = "cfg/config.json") -> Any:
    """Get a configuration value with hierarchical keys.
    
    This is a convenience function that maintains backward compatibility.
    """
    ConfigManager.load(config_path)
    return ConfigManager.get(*keys, default=default)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.DEBUG)
    
    try:
        # Load configuration
        config = ConfigManager.load("cfg/config.json")
        print(f"✅ Configuration loaded successfully")
        
        # Test hierarchical access
        raw_data = ConfigManager.get("paths", "raw_data")
        print(f"Raw data path: {raw_data}")
        
        # Test defaults
        custom_key = ConfigManager.get("nonexistent", "key", default="fallback")
        print(f"Nonexistent key (with default): {custom_key}")
        
        # Test component-specific configs
        validator_config = ConfigManager.get_processing_config("validator")
        print(f"Validator config: {validator_config}")
        
        # Test validation
        required = ["paths.raw_data", "llm_settings.model_name"]
        is_valid = ConfigManager.validate_required_keys(required)
        print(f"Configuration valid: {is_valid}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
