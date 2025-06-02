import hashlib
from abc import ABC, abstractmethod
from typing import Optional
import os

class EngineLM(ABC):
    system_prompt: str = "You are a helpful, creative, and smart assistant."
    model_string: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    cache_dir: Optional[str] = None

    def __init__(
        self,
        model_string: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        self.model = model_string
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache_dir = cache_dir

    @abstractmethod
    def generate(self, prompt, system_prompt=None, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass
    def _generate_cache_key(self, prompt: str) -> str:      
        """Generate a cache key based on the prompt."""
        # Use SHA-256 to create a unique hash of the prompt
        return hashlib.sha256(prompt.encode('utf-8')).hexdigest()
    def _get_cache_path(self, cache_key: str) -> str:
        """Get the file path for the cache based on the cache key."""
        if self.cache_dir is None:
            raise ValueError("Cache directory is not set.")
        return f"{self.cache_dir}/{cache_key}.json"
    def _load_from_cache(self, cache_key: str):
        """Load the response from cache if it exists."""
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            return None
    def _save_to_cache(self, cache_key: str, response: str):
        """Save the response to cache."""
        cache_path = self._get_cache_path(cache_key)
        with open(cache_path, 'w') as f:
            f.write(response)
    def _clear_cache(self):
        """Clear the cache directory."""
        if self.cache_dir is None:
            raise ValueError("Cache directory is not set.")
        import os
        for filename in os.listdir(self.cache_dir):
            file_path = os.path.join(self.cache_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
    def _set_cache_dir(self, cache_dir: str):
        """Set the cache directory."""
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        elif not os.path.isdir(cache_dir):
            raise ValueError(f"{cache_dir} is not a directory.")
        print(f"Cache directory set to: {self.cache_dir}")
    def _get_cache_dir(self) -> Optional[str]:
        """Get the current cache directory."""
        return self.cache_dir if self.cache_dir else None
    def _is_cache_enabled(self) -> bool:
        """Check if caching is enabled."""
        return self.cache_dir is not None and os.path.exists(self.cache_dir)
    def _disable_cache(self):   
        """Disable caching."""
        self.cache_dir = None
        print("Caching has been disabled.")
    def _enable_cache(self, cache_dir: str):
        """Enable caching with the specified directory."""
        self._set_cache_dir(cache_dir)
        print(f"Caching has been enabled with directory: {self.cache_dir}")
    def _get_model_string(self) -> str:
        """Get the model string."""
        return self.model_string if self.model_string else "No model string set."
    def _set_model_string(self, model_string: str):
        """Set the model string."""
        self.model_string = model_string
        print(f"Model string set to: {self.model_string}")
    def _get_temperature(self) -> float:
        """Get the temperature setting."""
        return self.temperature if self.temperature is not None else 0.7
    def _set_temperature(self, temperature: float):
        """Set the temperature for generation."""
        if not (0 <= temperature <= 1):
            raise ValueError("Temperature must be between 0 and 1.")
        self.temperature = temperature
        print(f"Temperature set to: {self.temperature}")
    def _get_max_tokens(self) -> Optional[int]:
        """Get the maximum tokens setting."""
        return self.max_tokens if self.max_tokens is not None else "No max tokens set."
    def _set_max_tokens(self, max_tokens: Optional[int]):
        """Set the maximum tokens for generation."""
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("Max tokens must be a positive integer.")
        self.max_tokens = max_tokens
        print(f"Max tokens set to: {self.max_tokens}")
    def _get_system_prompt(self) -> str:
        """Get the system prompt."""
        return self.system_prompt if self.system_prompt else "No system prompt set."
    def _set_system_prompt(self, system_prompt: str):
        """Set the system prompt."""
        if not system_prompt:
            raise ValueError("System prompt cannot be empty.")
        self.system_prompt = system_prompt
        print(f"System prompt set to: {self.system_prompt}")
    def _get_model_info(self) -> str:
        """Get information about the model."""
        return f"Model: {self.model_string}, Temperature: {self.temperature}, Max Tokens: {self.max_tokens}, Cache Directory: {self.cache_dir if self.cache_dir else 'Not set'}"
    def _get_engine_name(self) -> str:
        """Get the name of the engine."""
        return self.__class__.__name__
    def _get_engine_version(self) -> str:
        """Get the version of the engine."""
        return "1.0.0"

