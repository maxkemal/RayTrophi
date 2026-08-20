"""The tool loop: user message -> model -> tools -> engine -> answer."""

import json
import logging
import re

from config import MAX_TOOL_ITERATIONS, MAX_HISTORY_MESSAGES
from prompts.system_prompt import SYSTEM_PROMPT


class Orchestrator:
    def __init__(self, provider, executor, report=None):
        self.provider = provider
        self.executor = executor
        self.report = report or (lambda kind, text: None)
        self.conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.metrics = {
            # ★ NOT "task_success". Whether the user's task succeeded is not
            # something this loop can measure, and a metric that reports it
            # anyway is the same instrument failure as a coverage number that
            # counts written summaries and gets read as correct ones.
            "ended_without_tool_call": False,
            "stopped_early": False,
            "nudges": 0,
            "tool_failures": 0,
            "empty_answer": False,
            "tool_calls": 0,
            "invalid_calls": 0,
            "tokens_used": 0,
            "human_hints": 0,
            "recipe_used": False,
            "verification_performed": False
        }

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

    # -- horizon --------------------------------------------------------------

    # ★★★ Measured 2026-08-19, qwen3:8b, seven-step task: it completed two steps,
    # then WROTE the next tool call as JSON in its answer and asked "would you
    # like to continue?". MAX_TOOL_ITERATIONS was 64 and never came near; the
    # model quit at call seven. Small models do not fail at calling a tool, they
    # fail at the HORIZON - each finished step reads to them as a turn boundary,
    # and they drop out of tool-calling into chat.
    #
    # Prompting alone did not hold, so the loop checks for the two signatures and
    # pushes back. Bounded on purpose: a model that genuinely needs the user must
    # still be able to stop, so after NUDGE_LIMIT the answer stands and
    # `stopped_early` records that it was cut short rather than finished.
    NUDGE_LIMIT = 3
    _CALL_IN_PROSE = re.compile(r'"method"\s*:\s*"[a-z_]+\.[a-z_]+"')
    _ASKS_PERMISSION = re.compile(
        r"(would you like|shall i|should i (?:proceed|continue)|"
        r"let me know if|do you want me to|ready to proceed|"
        r"proceed with step|next,? (?:let'?s|we)|devam edeyim mi)", re.I)

    def _continuation_nudge(self, answer, reasoning=""):
        """Return a corrective message, or None to let the answer stand."""
        if self.metrics["nudges"] >= self.NUDGE_LIMIT:
            self.metrics["stopped_early"] = True
            return None
        if not answer.strip():
            # Thought, then said nothing. Nudging is the whole point here:
            # without it the turn ends as a silent no-op.
            if not reasoning.strip():
                return None
            self.metrics["stopped_early"] = True
            return ("You produced internal reasoning but no answer and no tool "
                    "call, so nothing happened. Take the next concrete step "
                    "now: make the tool call, or state your finding in plain "
                    "text. Do not answer with thinking alone.")
        described = bool(self._CALL_IN_PROSE.search(answer))
        asked = bool(self._ASKS_PERMISSION.search(answer))
        if not (described or asked):
            return None
        reason = ("You wrote a tool call as text instead of making it."
                  if described else
                  "You stopped to ask permission in the middle of a task.")
        return (reason + " Continue now without asking: the user already asked "
                "for every step of this task, so finishing it is not a new "
                "request. Make the calls yourself, one at a time, and check "
                "each result. Stop only when every step is done, or when you "
                "are genuinely blocked - and if you are blocked, say exactly "
                "what blocked you.")

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
            reasoning = response.get("reasoning") or ""
            raw_message = response.get("raw_message")

            if not tool_calls:
                self.conversation_history.append(
                    {"role": "assistant", "content": text_response})
                nudge = self._continuation_nudge(text_response, reasoning)
                if nudge:
                    self.metrics["nudges"] += 1
                    self.report("activity", "agent produced no usable step - "
                                            "nudged to continue")
                    self.conversation_history.append(
                        {"role": "user", "content": nudge})
                    continue
                self.metrics["ended_without_tool_call"] = True
                logging.info("Metrics for turn: %s", self.metrics)
                if text_response.strip():
                    return text_response
                # ★★★ NOT "Done.". An empty answer means the model said
                # nothing, and reporting that as completion is the same
                # instrument failure as a probe that answers 0 when it measured
                # nothing. Say what actually happened.
                self.metrics["empty_answer"] = True
                if reasoning.strip():
                    self.report("error", "model returned reasoning but no answer")
                    return ("The model finished its turn without producing an "
                            "answer or a tool call - only internal reasoning. "
                            "Nothing was reported and the task may be "
                            "unfinished. Ask it to continue.")
                return ("The model returned an empty response. Nothing was done "
                        "and nothing was reported.")

            if text_response.strip():
                self.report("thought", text_response.strip())

            self.conversation_history.append(raw_message or {
                "role": "assistant",
                "content": text_response,
                "tool_calls": tool_calls,
            })

            for call in tool_calls:
                self.metrics["tool_calls"] += 1
                if call["name"] == "search_capabilities":
                    self.metrics["recipe_used"] = True
                elif call["name"] == "get_scene_context":
                    self.metrics["verification_performed"] = True

                try:
                    result = self.executor.execute(call["name"], call.get("args", {}) or {})
                except Exception as exc:  # noqa: BLE001
                    # A tool crash must reach the model as a result it can react
                    # to, not kill the turn.
                    logging.exception("tool %s raised", call.get("name"))
                    result = {"ok": False, "error": "tool raised: %s" % exc}
                
                if isinstance(result, dict) and not result.get("ok", True):
                    self.metrics["invalid_calls"] += 1

                self.conversation_history.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": call["name"],
                    "content": json.dumps(result, default=str),
                })

        self.report("error", "hit the %d step limit" % MAX_TOOL_ITERATIONS)
        logging.info("Metrics for turn: %s", self.metrics)
        return ("I reached the %d-step limit for one request. Here is where I stopped - "
                "ask me to continue if that looks right." % MAX_TOOL_ITERATIONS)
