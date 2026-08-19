import logging

from .openai_provider import HAS_OPENAI, OpenAIProvider


class LocalLLMProvider(OpenAIProvider):
    """Any local model that speaks the OpenAI chat API.

    Ollama:   http://localhost:11434/v1
    LM Studio: http://localhost:1234/v1

    Tool calling has to be supported by the model itself; a model without it
    will answer in prose and never touch the engine.
    """

    def __init__(self, base_url, model="local-model"):
        if not HAS_OPENAI:
            raise ImportError("openai is not installed. Run 'pip install openai'")
        from openai import OpenAI
        # Local servers ignore the key but the SDK requires a non-empty string.
        self.client = OpenAI(api_key="local", base_url=base_url)
        self.model = model
        logging.info("LocalLLMProvider ready (%s, model %s)", base_url, self.model)
