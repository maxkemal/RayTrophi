"""RayTrophi agent runtime.

Two pipe connections on purpose:

  * `chat` - polled once a second from the main thread. Polling IS the panel's
    heartbeat, so it must never be blocked by model work. When one connection
    did both, a long model turn read as "agent disconnected", the panel offered
    a Start button, and a second agent process appeared.
  * `tools` - owned by the worker thread that runs the model's tool calls.

The engine serves several pipe instances, so both connect at once and your
PowerShell/pytest session can still attach alongside.
"""

import logging
import logging.handlers
import os
import queue
import sys
import threading
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (GEMINI_API_KEY, GEMINI_MODEL, IPC_PIPE_NAME, LLM_PROVIDER, LOCAL_LLM_MODEL,
                    LOCAL_LLM_URL, OPENAI_API_KEY, OPENAI_MODEL, POLL_INTERVAL_SEC)
from core.ipc_client import IPCClient
from core.orchestrator import Orchestrator
from core.registry import CapabilityRegistry
from core.session import SessionManager
from core.tool_executor import ToolExecutor
from providers.gemini_provider import GeminiProvider
from providers.local_provider import LocalLLMProvider
from providers.openai_provider import MockProvider, OpenAIProvider
from providers.anthropic_provider import AnthropicProvider

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=[logging.handlers.RotatingFileHandler("agent.log", maxBytes=5*1024*1024, backupCount=3, encoding="utf-8"),
              logging.StreamHandler()])

SENDER = "RayTrophi Agent"


def build_provider():
    from config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL
    
    if LLM_PROVIDER == "openai":
        return OpenAIProvider(api_key=OPENAI_API_KEY, model=OPENAI_MODEL)
    if LLM_PROVIDER == "gemini":
        return GeminiProvider(api_key=GEMINI_API_KEY, model=GEMINI_MODEL)
    if LLM_PROVIDER == "anthropic":
        return AnthropicProvider(api_key=ANTHROPIC_API_KEY, model=ANTHROPIC_MODEL)
    if LLM_PROVIDER == "local":
        return LocalLLMProvider(base_url=LOCAL_LLM_URL, model=LOCAL_LLM_MODEL)
    return MockProvider()


def main():
    session = SessionManager().get_session()

    chat_ipc = IPCClient(pipe_name=IPC_PIPE_NAME, agent_id=session.agent_id,
                         session_id=session.session_id, label="chat")
    tool_ipc = IPCClient(pipe_name=IPC_PIPE_NAME, agent_id=session.agent_id,
                         session_id=session.session_id, label="tools")

    logging.info("connecting to RayTrophi as %s (session %s)...",
                 session.agent_id, session.session_id)
    if not chat_ipc.connect(timeout_sec=10) or not tool_ipc.connect(timeout_sec=10):
        logging.error("could not connect to the engine; exiting.")
        return 1

    registry = CapabilityRegistry()
    if not registry.bootstrap(tool_ipc):
        logging.warning("registry bootstrap failed; method validation is disabled "
                        "and every call goes straight to the engine.")

    # Outbound chat goes through one queue drained by the main thread, so the
    # chat connection stays single-threaded while the worker reports freely.
    outbox = queue.Queue()
    prompts = queue.Queue()

    def report(kind, text):
        outbox.put((kind, text))

    executor = ToolExecutor(ipc=tool_ipc, registry=registry, report=report)
    provider = build_provider()
    provider.report = report
    orchestrator = Orchestrator(provider=provider, executor=executor,
                                report=report)

    stop = threading.Event()

    def worker():
        while not stop.is_set():
            try:
                prompt = prompts.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                answer = orchestrator.handle_user_message(prompt)
                outbox.put(("reply", answer))
            except Exception as exc:  # noqa: BLE001 - one bad turn must not end the run
                logging.exception("turn failed")
                outbox.put(("error", "That request failed inside the agent: %s" % exc))
            finally:
                prompts.task_done()

    worker_thread = threading.Thread(target=worker, name="agent-worker", daemon=True)
    worker_thread.start()

    identity = registry.identity or {}
    outbox.put(("reply", "%s connected to %s %s. %d methods available."
                % (session.agent_id, identity.get("app", "RayTrophi"),
                   identity.get("version", ""), len(registry.methods))))

    logging.info("entering poll loop")
    try:
        while True:
            polled = chat_ipc.call("agent.chat_poll", {"agent_id": session.agent_id})
            if "error" in polled:
                logging.warning("poll failed: %s", polled["error"])
                # Only a dropped connection needs reconnecting; a dispatch error
                # is answered on a healthy pipe and must not tear it down.
                if not chat_ipc.connected and not chat_ipc.connect(timeout_sec=10):
                    logging.error("chat connection lost and could not be restored.")
                    break
                time.sleep(POLL_INTERVAL_SEC)
                continue

            for prompt in polled.get("result", {}).get("prompts", []):
                content = prompt.get("content", "")
                if not content:
                    continue
                logging.info("prompt: %s", content)
                if not prompts.empty():
                    outbox.put(("activity", "queued; finishing the current request first"))
                prompts.put(content)

            while True:
                try:
                    kind, text = outbox.get_nowait()
                except queue.Empty:
                    break
                sent = chat_ipc.call("agent.chat_send",
                                     {"sender": SENDER, "content": text, "type": kind})
                if "error" in sent:
                    logging.warning("chat_send failed (%s): %s", kind, sent["error"])

            time.sleep(POLL_INTERVAL_SEC)

    except KeyboardInterrupt:
        logging.info("stopping.")
    finally:
        stop.set()
        chat_ipc.close()
        tool_ipc.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
