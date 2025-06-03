import os
from openai import OpenAI
import platformdirs
from typing import List, Union, Optional
from .base import EngineLM

class ChatLlama3(EngineLM):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, honest, and precise assistant."
    BASE_URL = "http://localhost:8000/v1"  # Assuming local deployment, adjust if needed

    def __init__(
        self,
        model_string: str = "llama-3-70b",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        cache_dir: Optional[str] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        api_key: str = "not-needed",  # Local deployment typically doesn't need API key
        **kwargs
    ):
        """
        Initializes an interface for interacting with Llama-3-70B model.

        :param model_string: The model identifier. Defaults to 'llama-3-70b'.
        :param temperature: Controls randomness in generation. Defaults to 0.7.
        :param max_tokens: Maximum tokens to generate. Optional.
        :param cache_dir: Directory for caching responses. Optional.
        :param system_prompt: The default system prompt for responses.
        :param api_key: API key for authentication (if needed).
        :param kwargs: Additional keyword arguments.
        """
        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_llama3_{model_string}.db")

        super().__init__(
            cache_path=cache_path,
            model_string=model_string,
            temperature=temperature,
            max_tokens=max_tokens,
            cache_dir=cache_dir
        )

        self.system_prompt = system_prompt
        
        # Initialize the client with local endpoint
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.BASE_URL
        )
        self.model_string = model_string

    def generate(self, content: Union[str, List[str]], system_prompt: str = None, **kwargs):
        """Generate response using Llama-3 model"""
        if isinstance(content, list):
            return self._generate_from_multiple_input(content, system_prompt=system_prompt, **kwargs)
        return self._generate_from_single_prompt(content, system_prompt=system_prompt, **kwargs)

    def _generate_from_single_prompt(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.95,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0
    ):
        """Generate response from a single prompt"""
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        
        # Check cache
        cache_or_none = self._check_cache(sys_prompt_arg + prompt)
        if cache_or_none is not None:
            return cache_or_none

        # Create messages
        messages = [
            {"role": "system", "content": sys_prompt_arg},
            {"role": "user", "content": prompt}
        ]

        try:
            # Call API with Llama-specific parameters
            response = self.client.chat.completions.create(
                model=self.model_string,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty
            )

            response_text = response.choices[0].message.content
            self._save_cache(sys_prompt_arg + prompt, response_text)
            return response_text
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return f"Error: {str(e)}"

    def _generate_from_multiple_input(
        self,
        contents: List[str],
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.95
    ):
        """Generate response from multiple inputs"""
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        
        # Create messages array
        messages = [{"role": "system", "content": sys_prompt_arg}]
        for content in contents:
            messages.append({"role": "user", "content": content})

        # Create cache key
        cache_key = sys_prompt_arg + "".join(contents)
        cache_or_none = self._check_cache(cache_key)
        if cache_or_none is not None:
            return cache_or_none

        try:
            # Call API
            response = self.client.chat.completions.create(
                model=self.model_string,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p
            )

            response_text = response.choices[0].message.content
            self._save_cache(cache_key, response_text)
            return response_text
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return f"Error: {str(e)}"

    def __call__(self, prompt, **kwargs):
        """Convenience method to support calling the instance directly"""
        return self.generate(prompt, **kwargs)