class LLMProvider:
    """Base class for all LLM providers."""
    
    def generate(self, messages: list, tools: list = None) -> dict:
        """
        Sends the conversation history and available tools to the LLM.
        
        Returns a dictionary containing:
        {
            "text": "The final response text (if no tools called)",
            "tool_calls": [
                {"id": "...", "name": "...", "args": {...}}
            ]
        }
        """
        raise NotImplementedError("LLMProvider subclasses must implement generate()")
