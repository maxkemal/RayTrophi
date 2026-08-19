"""The tool loop: user message -> model -> tools -> engine -> answer."""

import json
import logging

from config import MAX_TOOL_ITERATIONS, MAX_HISTORY_MESSAGES
from prompts.system_prompt import SYSTEM_PROMPT


class Orchestrator:
    def __init__(self, provider, executor, report=None):
        self.provider = provider
        self.executor = executor
        self.report = report or (lambda kind, text: None)
        self.conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]

    # -- history ------------------------------------------------------------

    def _trim_history(self):
        """Keep the system prompt and the most recent exchanges.

        Without this a long session grows until the provider refuses it, and
        the failure arrives as an opaque provider error in the middle of a task.
        """
        if len(self.conversation_history) <= MAX_HISTORY_MESSAGES:
            return
        system = self.conversation_history[0]
        tail = self.conversation_history[-(MAX_HISTORY_MESSAGES - 1):]
        # Never start the tail with an orphan tool result: providers reject a
        # tool message whose originating call is no longer in the history.
        while tail and tail[0].get("role") == "tool":
            tail.pop(0)
        self.conversation_history = [system] + tail

    # -- main loop ----------------------------------------------------------

    def handle_user_message(self, text):
        logging.info("user: %s", text)
        self.conversation_history.append({"role": "user", "content": text})

        for iteration in range(MAX_TOOL_ITERATIONS):
            self._trim_history()
            response = self.provider.generate(
                messages=list(self.conversation_history),
                tools=self.executor.get_schemas())

            if "error" in response:
                logging.error("provider error: %s", response["error"])
                self.report("error", "model call failed: %s" % response["error"])
                return "The model call failed: %s" % response["error"]

            tool_calls = response.get("tool_calls", [])
            text_response = response.get("text") or ""
            raw_message = response.get("raw_message")

            if not tool_calls:
                self.conversation_history.append(
                    {"role": "assistant", "content": text_response})
                return text_response or "Done."

            if text_response.strip():
                self.report("thought", text_response.strip())

            self.conversation_history.append(raw_message or {
                "role": "assistant",
                "content": text_response,
                "tool_calls": tool_calls,
            })

            for call in tool_calls:
                try:
                    result = self.executor.execute(call["name"], call.get("args", {}) or {})
                except Exception as exc:  # noqa: BLE001
                    # A tool crash must reach the model as a result it can react
                    # to, not kill the turn.
                    logging.exception("tool %s raised", call.get("name"))
                    result = {"ok": False, "error": "tool raised: %s" % exc}
                self.conversation_history.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": call["name"],
                    "content": json.dumps(result, default=str),
                })

        self.report("error", "hit the %d step limit" % MAX_TOOL_ITERATIONS)
        return ("I reached the %d-step limit for one request. Here is where I stopped - "
                "ask me to continue if that looks right." % MAX_TOOL_ITERATIONS)
