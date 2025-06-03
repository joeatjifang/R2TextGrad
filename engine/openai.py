try:
    from openai import OpenAI, AzureOpenAI
except ImportError:
    raise ImportError("If you'd like to use OpenAI models, please install the openai package by running `pip install openai`, and add 'OPENAI_API_KEY' to your environment variables.")

import os
import json
import base64
import platformdirs
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from typing import List, Union, Optional

from .base import EngineLM
from .engine_utils import get_image_type_from_bytes

# Default base URL for OLLAMA
OLLAMA_BASE_URL = 'http://localhost:11434/v1'

# Check if the user set the OLLAMA_BASE_URL environment variable
if os.getenv("OLLAMA_BASE_URL"):
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

class OpenAIEmbedding:
    def __init__(self, api_key: str = None):
        """
        Initialize OpenAI embedding client
        
        Args:
            api_key: OpenAI API key. If not provided, will use environment variable.
        """
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )

    def get_embedding(self, text: str, model: str = "text-embedding-ada-002") -> List[float]:
        """
        Get embeddings for input text using OpenAI's embedding models
        
        Args:
            text: Input text to embed
            model: Model to use for embeddings. Defaults to text-embedding-ada-002
            
        Returns:
            List of floats representing the text embedding
            
        Raises:
            Exception: If embedding generation fails
        """
        try:
            response = self.client.embeddings.create(
                model=model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Failed to get embedding: {e}")
            raise

class ChatOpenAI(EngineLM):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        cache_dir: Optional[str] = None,
        seed: Optional[int] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool = False,
        base_url: str = None,
        **kwargs):
        """
        :param model_string:
        :param system_prompt:
        :param base_url: Used to support Ollama
        """
        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_openai_{model_string}.db")

        super().__init__(cache_path=cache_path, model_string=model_string, temperature=temperature, max_tokens=max_tokens, cache_dir=cache_dir)

        self.system_prompt = system_prompt
        self.base_url = base_url
        self.seed = seed
        
        if not base_url:
            if os.getenv("OPENAI_API_KEY") is None:
                raise ValueError("Please set the OPENAI_API_KEY environment variable if you'd like to use OpenAI models.")
            
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
        elif base_url and base_url == OLLAMA_BASE_URL:
            self.client = OpenAI(
                base_url=base_url,
                api_key="ollama"
            )
        else:
            raise ValueError("Invalid base URL provided. Please use the default OLLAMA base URL or None.")

        self.model_string = model_string
        self.is_multimodal = is_multimodal

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def generate(self, content: Union[str, List[Union[str, bytes]]], system_prompt: str=None, **kwargs):
        if isinstance(content, str):
            return self._generate_from_single_prompt(content, system_prompt=system_prompt, **kwargs)
        
        elif isinstance(content, list):
            has_multimodal_input = any(isinstance(item, bytes) for item in content)
            if (has_multimodal_input) and (not self.is_multimodal):
                raise NotImplementedError("Multimodal generation is only supported for Claude-3 and beyond.")
            
            return self._generate_from_multiple_input(content, system_prompt=system_prompt, **kwargs)

    def _generate_from_single_prompt(
        self, prompt: str, system_prompt: str=None, temperature=0, max_tokens=2000, top_p=0.99
    ):

        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt

        cache_or_none = self._check_cache(sys_prompt_arg + prompt)
        if cache_or_none is not None:
            return cache_or_none

        response = self.client.chat.completions.create(
            model=self.model_string,
            messages=[
                {"role": "system", "content": sys_prompt_arg},
                {"role": "user", "content": prompt},
            ],
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        response = response.choices[0].message.content
        self._save_cache(sys_prompt_arg + prompt, response)
        return response

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

    def _format_content(self, content: List[Union[str, bytes]]) -> List[dict]:
        """Helper function to format a list of strings and bytes into a list of dictionaries to pass as messages to the API.
        """
        formatted_content = []
        for item in content:
            if isinstance(item, bytes):
                # For now, bytes are assumed to be images
                image_type = get_image_type_from_bytes(item)
                base64_image = base64.b64encode(item).decode('utf-8')
                formatted_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_type};base64,{base64_image}"
                    }
                })
            elif isinstance(item, str):
                formatted_content.append({
                    "type": "text",
                    "text": item
                })
            else:
                raise ValueError(f"Unsupported input type: {type(item)}")
        return formatted_content

    def _generate_from_multiple_input(
        self, content: List[Union[str, bytes]], system_prompt=None, temperature=0, max_tokens=2000, top_p=0.99
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        formatted_content = self._format_content(content)

        cache_key = sys_prompt_arg + json.dumps(formatted_content)
        cache_or_none = self._check_cache(cache_key)
        if cache_or_none is not None:
            return cache_or_none

        response = self.client.chat.completions.create(
            model=self.model_string,
            messages=[
                {"role": "system", "content": sys_prompt_arg},
                {"role": "user", "content": formatted_content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        response_text = response.choices[0].message.content
        self._save_cache(cache_key, response_text)
        return response_text

