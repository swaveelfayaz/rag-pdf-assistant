from typing import Protocol, List, Dict
import litellm

from utils.config import config
from utils.exceptions import GenerationError
from utils.logger import get_logger

log = get_logger(__name__)

class LLMBackend(Protocol):
    def generate(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> tuple[str, int]:
        """
        Generate a response given a list of messages.
        Returns a tuple of (answer_text, tokens_used).
        """
        ...
        
    def generate_stream(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int):
        """
        Stream a response given a list of messages.
        Yields chunk_text string.
        """
        ...

class LiteLLMBackend:
    def __init__(self):
        self.model = config.litellm_model
        log.info("LiteLLM Backend initialised (model: %s).", self.model)
        
        # If using Ollama, we need to pass the api_base
        self.api_base = None
        if self.model.startswith("ollama/"):
            self.api_base = config.ollama_base_url
            
    def generate(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> tuple[str, int]:
        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                api_base=self.api_base
            )
            answer_text = response.choices[0].message.content.strip()
            tokens_used = response.usage.total_tokens if hasattr(response, 'usage') and response.usage else 0
            return answer_text, tokens_used
        except Exception as exc:
            raise GenerationError(
                f"LiteLLM completion failed for model {self.model}: {exc}."
            ) from exc

    def generate_stream(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int):
        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                api_base=self.api_base,
                stream=True
            )
            
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as exc:
            raise GenerationError(
                f"LiteLLM streaming failed for model {self.model}: {exc}."
            ) from exc


_backend_instance: LLMBackend | None = None

def get_llm_backend() -> LLMBackend:
    """Lazily load the LLM backend."""
    global _backend_instance
    if _backend_instance is None:
        _backend_instance = LiteLLMBackend()
    return _backend_instance
