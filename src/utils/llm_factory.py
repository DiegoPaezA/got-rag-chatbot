import os
import logging
from typing import Any, Dict
from enum import Enum

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """
    Enumeration of supported LLM providers.
    """
    GOOGLE = "google"
    OPENAI = "openai"

class LLMFactory:
    """
    Factory class to create and configure LangChain Chat Model instances.
    
    This class centralizes the initialization logic, allowing the application
    to switch between providers (e.g., Google Gemini, OpenAI) based on configuration.
    """

    @staticmethod
    def create_llm(
        config: Dict[str, Any], 
        provider: str = "google"
    ) -> BaseChatModel:
        """
        Creates and returns a configured LangChain Chat Model.

        Args:
            config (Dict[str, Any]): A dictionary containing model settings 
                                     (e.g., model_name, temperature).
            provider (str): The provider name (default: "google").

        Returns:
            BaseChatModel: An initialized LangChain chat model.

        Raises:
            ValueError: If the provider is not supported or API keys are missing.
        """
        try:
            # Normalize provider string to match Enum
            provider_enum = LLMProvider(provider.lower())
        except ValueError:
            valid_options = [p.value for p in LLMProvider]
            raise ValueError(f"Unsupported provider '{provider}'. Valid options: {valid_options}")

        model_hint = config.get("model_name") or config.get("model")
        logger.info(f"Initializing LLM with provider: {provider_enum.value} | Model: {model_hint}")

        if provider_enum == LLMProvider.GOOGLE:
            return LLMFactory._create_google_model(config)
        
        elif provider_enum == LLMProvider.OPENAI:
            return LLMFactory._create_openai_model(config)
        
        raise ValueError(f"Provider '{provider}' is technically valid but not implemented.")

    @staticmethod
    def _create_google_model(config: Dict[str, Any]) -> ChatGoogleGenerativeAI:
        """
        Internal helper to create a Google Gemini Chat model.

        Args:
            config (Dict[str, Any]): Configuration dictionary with 'model_name', 'temperature', etc.

        Returns:
            ChatGoogleGenerativeAI: The configured Gemini client.
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("GOOGLE_API_KEY not found in environment variables.")
            raise ValueError("GOOGLE_API_KEY is missing. Please set it in your .env file.")

        # Extract parameters with safe defaults
        model_name = config.get("model_name") or config.get("model") or "gemini-2.5-flash"
        temperature = config.get("temperature", 0.0)
        max_retries = config.get("max_retries", 3)

        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_retries=max_retries,
            google_api_key=api_key,
        )

    @staticmethod
    def _create_openai_model(config: Dict[str, Any]) -> ChatOpenAI:
        """
        Internal helper to create an OpenAI Chat model.

        Args:
            config (Dict[str, Any]): Configuration dictionary with 'model_name', 'temperature', etc.

        Returns:
            ChatOpenAI: The configured OpenAI client.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY not found in environment variables.")
            raise ValueError("OPENAI_API_KEY is missing. Please set it in your .env file.")

        # Extract parameters with safe defaults for OpenAI
        # Defaulting to gpt-4o-mini as it is cost-effective and capable
        model_name = config.get("model_name") or config.get("model") or "gpt-4o-mini"
        temperature = config.get("temperature", 0.0)
        max_retries = config.get("max_retries", 3)

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_retries=max_retries,
            api_key=api_key
        )