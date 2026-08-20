import json
import logging

from .base import LLMProvider

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

class AnthropicProvider(LLMProvider):
    def __init__(self, api_key, model="claude-3-5-sonnet-20240620"):
        if not HAS_ANTHROPIC:
            raise ImportError("anthropic is not installed. Run 'pip install anthropic'")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        logging.info("AnthropicProvider ready (model %s)", self.model)

    @staticmethod
    def _tool_declarations(tools):
        declarations = []
        for tool in tools or []:
            if tool.get("type") != "function":
                continue
            function = tool["function"]
            declarations.append({
                "name": function["name"],
                "description": function.get("description", ""),
                "input_schema": function.get("parameters", {}),
            })
        return declarations

    def _to_anthropic_messages(self, messages):
        system_instruction = None
        anthropic_messages = []
        for message in messages:
            role = message.get("role")
            
            if role == "system":
                system_instruction = message.get("content") or ""
                continue
                
            if role == "user":
                anthropic_messages.append({"role": "user", "content": str(message.get("content") or "")})
                continue
                
            if role == "assistant":
                content = []
                if message.get("content"):
                    content.append({"type": "text", "text": str(message.get("content"))})
                for call in message.get("tool_calls") or []:
                    content.append({
                        "type": "tool_use",
                        "id": call["id"],
                        "name": call["name"],
                        "input": call.get("args", {})
                    })
                if content:
                    anthropic_messages.append({"role": "assistant", "content": content})
                continue
                
            if role == "tool":
                payload = message.get("content")
                if isinstance(payload, dict):
                    payload = json.dumps(payload)
                else:
                    payload = str(payload)
                anthropic_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": message.get("tool_call_id"),
                        "content": payload
                    }]
                })
        return system_instruction, anthropic_messages

    def generate(self, messages, tools=None):
        try:
            system_instruction, anthropic_messages = self._to_anthropic_messages(messages)
            
            kwargs = {
                "model": self.model,
                "max_tokens": 4096,
                "messages": anthropic_messages
            }
            if system_instruction:
                kwargs["system"] = system_instruction
                
            declarations = self._tool_declarations(tools)
            if declarations:
                kwargs["tools"] = declarations
                
            response = self.client.messages.create(**kwargs)
            
            text_blocks = [b.text for b in response.content if b.type == "text"]
            text = "\n".join(text_blocks)
            
            tool_calls = []
            for b in response.content:
                if b.type == "tool_use":
                    tool_calls.append({
                        "id": b.id,
                        "name": b.name,
                        "args": b.input
                    })
                    
            return {
                "text": text,
                "tool_calls": tool_calls,
                "raw_message": {"role": "assistant", "content": text, "tool_calls": tool_calls}
            }
            
        except Exception as exc:
            logging.error("AnthropicProvider error: %s", exc)
            return {"error": str(exc)}
