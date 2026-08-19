import json
import logging

from .base import LLMProvider

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key, model="gpt-4o-mini"):
        if not HAS_OPENAI:
            raise ImportError("openai is not installed. Run 'pip install openai'")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        logging.info("OpenAIProvider ready (model %s)", self.model)

    def generate(self, messages, tools=None):
        try:
            kwargs = {"model": self.model, "messages": messages}
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            response = self.client.chat.completions.create(**kwargs)
            message = response.choices[0].message

            tool_calls = []
            for call in message.tool_calls or []:
                try:
                    args = json.loads(call.function.arguments or "{}")
                except ValueError as exc:
                    # Malformed arguments are the model's mistake to fix, so
                    # they travel on as a tool result rather than crashing here.
                    logging.warning("unparseable tool arguments for %s: %s",
                                    call.function.name, exc)
                    args = {"__parse_error": str(exc),
                            "__raw": call.function.arguments}
                tool_calls.append({"id": call.id, "name": call.function.name,
                                   "args": args})

            return {
                "text": message.content or "",
                "tool_calls": tool_calls,
                "raw_message": message,   # appended verbatim to the history
            }

        except Exception as exc:  # noqa: BLE001
            logging.error("OpenAIProvider error: %s", exc)
            return {"error": str(exc)}


class MockProvider(LLMProvider):
    """No model behind it. Exercises the tool loop and the panel end to end."""

    def __init__(self):
        logging.warning("MockProvider active - no real model. Set LLM_PROVIDER and a key "
                        "in RayTrophiAgent/.env for real work.")

    def generate(self, messages, tools=None):
        if messages and messages[-1].get("role") == "tool":
            return {"text": "Mock mode: the tool call went through and returned a result. "
                            "Configure a provider in .env for real reasoning."}

        last_user = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                last_user = str(message.get("content") or "").lower()
                break

        if any(word in last_user for word in ("scene", "state", "what is", "describe")):
            return {"text": "", "tool_calls": [
                {"id": "mock_context", "name": "get_scene_context", "args": {}}]}
        if any(word in last_user for word in ("fire", "burn", "flame", "yak", "ate")):
            return {"text": "", "tool_calls": [
                {"id": "mock_search", "name": "search_capabilities", "args": {"query": "fire"}}]}
        return {"text": "Mock provider: ask about the scene, or about fire, to exercise a "
                        "tool call. Configure a real provider in .env for anything else."}
