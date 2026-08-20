"""Gemini provider.

★ The tool round trip has to survive the translation. The first version turned
every history entry into plain text, so the model never saw its OWN function
calls - only their results. A model that cannot see what it just called repeats
the call, which looks like the engine ignoring it.
"""

import json
import logging

from .base import LLMProvider

try:
    from google import genai
    from google.genai import types
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False


class GeminiProvider(LLMProvider):
    def __init__(self, api_key, model="gemini-2.5-flash"):
        if not HAS_GEMINI:
            raise ImportError("google-genai is not installed. Run 'pip install google-genai'")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        logging.info("GeminiProvider ready (model %s)", self.model)

    # -- conversion ---------------------------------------------------------

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
                "parameters": function.get("parameters", {}),
            })
        return declarations

    def _to_contents(self, messages):
        """OpenAI-shaped history -> Gemini contents; system prompt returned apart."""
        system_instruction = None
        contents = []
        for message in messages:
            role = message.get("role")

            if role == "system":
                system_instruction = message.get("content") or None
                continue

            if role == "user":
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=str(message.get("content") or ""))]))
                continue

            if role == "assistant":
                parts = []
                text = message.get("content")
                if text:
                    parts.append(types.Part.from_text(text=str(text)))
                for call in message.get("tool_calls") or []:
                    part = types.Part.from_function_call(
                        name=call["name"], args=call.get("args") or {})
                    if call.get("thought_signature"):
                        part.thought_signature = call["thought_signature"]
                    if call.get("thought"):
                        part.thought = call["thought"]
                    parts.append(part)
                if parts:
                    contents.append(types.Content(role="model", parts=parts))
                continue

            if role == "tool":
                payload = message.get("content")
                try:
                    payload = json.loads(payload) if isinstance(payload, str) else payload
                except ValueError:
                    payload = {"result": payload}
                if not isinstance(payload, dict):
                    payload = {"result": payload}
                    
                tool_parts = []
                if "image_base64" in payload:
                    import base64
                    img_data = base64.b64decode(payload.pop("image_base64"))
                    tool_parts.append(types.Part.from_bytes(data=img_data, mime_type="image/jpeg"))

                tool_parts.append(types.Part.from_function_response(
                    name=message.get("name", "tool"), response=payload))
                
                contents.append(types.Content(
                    role="user",
                    parts=tool_parts))
        return system_instruction, contents

    @staticmethod
    def _text_of(response):
        """Safe text extraction: a tool-only reply has no text parts at all."""
        chunks = []
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                if getattr(part, "text", None):
                    chunks.append(part.text)
        return "".join(chunks)

    # -- generate -----------------------------------------------------------

    def generate(self, messages, tools=None):
        import time
        for attempt in range(3):
            try:
                system_instruction, contents = self._to_contents(messages)
                config = {}
                if system_instruction:
                    config["system_instruction"] = system_instruction
                declarations = self._tool_declarations(tools)
                if declarations:
                    config["tools"] = [{"function_declarations": declarations}]

                response = self.client.models.generate_content(
                    model=self.model, contents=contents, config=config)

                text = self._text_of(response)
                tool_calls = []
                index = 0
                for candidate in getattr(response, "candidates", None) or []:
                    content = getattr(candidate, "content", None)
                    for part in getattr(content, "parts", None) or []:
                        call = getattr(part, "function_call", None)
                        if call:
                            tool_calls.append({
                                "id": "call_%d_%s" % (index, call.name),
                                "name": call.name,
                                "args": dict(call.args or {}),
                                "thought_signature": getattr(part, "thought_signature", None),
                                "thought": getattr(part, "thought", None)
                            })
                            index += 1

                return {
                    "text": text,
                    "tool_calls": tool_calls,
                    # Normalised, not the SDK object: the orchestrator feeds this
                    # back through _to_contents on the next turn.
                    "raw_message": {"role": "assistant", "content": text,
                                    "tool_calls": tool_calls},
                }

            except Exception as exc:  # noqa: BLE001
                err_str = str(exc)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    delay = 40
                    msg = "Rate limit (429) hit. Sleeping %ds before retry %d/3..." % (delay, attempt + 1)
                    logging.warning(msg)
                    if getattr(self, "report", None):
                        self.report("activity", msg)
                    time.sleep(delay)
                    continue
                logging.error("GeminiProvider error: %s", exc)
                return {"error": err_str}
        return {"error": "Exceeded maximum retries due to rate limit (429 RESOURCE_EXHAUSTED)"}
