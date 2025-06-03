import os
from openai import OpenAI
import platformdirs
from typing import List, Union, Optional
from .base import EngineLM

class ChatDeepSeek(EngineLM):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."
    ALIYUN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(
        self,
        model_string: str = "deepseek-v3",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        cache_dir: Optional[str] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        **kwargs
    ):
        """
        Initializes an interface for interacting with Aliyun's DeepSeek models.

        :param model_string: The model identifier for DeepSeek. Defaults to 'deepseek-r1'.
        :param system_prompt: The default system prompt to use when generating responses.
        :param kwargs: Additional keyword arguments to pass to the constructor.

        Environment variables:
        - DASHSCOPE_API_KEY: The API key for authenticating with Aliyun DashScope.

        Raises:
            ValueError: If the DASHSCOPE_API_KEY environment variable is not set.
        """
        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_deepseek_{model_string}.db")

        super().__init__(
            cache_path=cache_path,
            model_string=model_string,
            temperature=temperature,
            max_tokens=max_tokens,
            cache_dir=cache_dir
        )

        self.system_prompt = system_prompt
        
        if os.getenv("DASHSCOPE_API_KEY") is None:
            raise ValueError(
                "Please set the DASHSCOPE_API_KEY environment variable to use DeepSeek models. "
                "You can get it from: https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key"
            )
        
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=self.ALIYUN_BASE_URL
        )
        self.model_string = model_string

    def generate(self, content: Union[str, List[str]], system_prompt: str = None, **kwargs):
        """Generate response using DeepSeek model"""
        if isinstance(content, list):
            return self._generate_from_multiple_input(content, system_prompt=system_prompt, **kwargs)
        return self._generate_from_single_prompt(content, system_prompt=system_prompt, **kwargs)

    def _generate_from_single_prompt(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0,
        max_tokens: int = 2000,
        top_p: float = 0.99
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

        # Call API
        response = self.client.chat.completions.create(
            model=self.model_string,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )

        response_text = response.choices[0].message.content
        self._save_cache(sys_prompt_arg + prompt, response_text)
        return response_text

    def _generate_from_multiple_input(
        self,
        contents: List[str],
        system_prompt: str = None,
        temperature: float = 0,
        max_tokens: int = 2000,
        top_p: float = 0.99
    ):
        """Generate response from multiple inputs"""
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        
        # Create messages array with alternating user/assistant roles
        messages = [{"role": "system", "content": sys_prompt_arg}]
        for content in contents:
            messages.append({"role": "user", "content": content})

        # Create cache key from all contents
        cache_key = sys_prompt_arg + "".join(contents)
        cache_or_none = self._check_cache(cache_key)
        if cache_or_none is not None:
            return cache_or_none

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

    def __call__(self, prompt, **kwargs):
        """Convenience method to support calling the instance directly"""
        return self.generate(prompt, **kwargs)