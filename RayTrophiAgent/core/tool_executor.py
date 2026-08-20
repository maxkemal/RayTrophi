"""The meta-tools the model is given, and how they reach the engine."""

import json
import logging
import os

META_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_capabilities",
            "description": ("Search RayTrophi for workflow recipes and methods that serve a "
                            "goal stated in plain language. Start here whenever you are not "
                            "certain a capability exists."),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "The goal, e.g. 'make a wooden object burn'"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_capability",
            "description": ("Get the exact parameter schema, types, defaults and required "
                            "security capability of one RPC method."),
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {"type": "string",
                               "description": "Exact method name, e.g. 'fluid.create_domain'"}
                },
                "required": ["method"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_scene_context",
            "description": ("Read the current state of the application: objects, lights, "
                            "camera, simulation domains, timeline frame, render and viewport "
                            "measurement. Use it before context-dependent work and again "
                            "afterwards to verify what actually changed."),
            "parameters": {
                "type": "object",
                "properties": {
                    "include_probe": {
                        "type": "boolean",
                        "description": ("Also measure the last viewport frame (mean luminance, "
                                        "black and NaN fractions). Requires frame capture.")
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "poll_rpc",
            "description": "Call an RPC method repeatedly until a specific key in its result matches a target value. Use this instead of looping yourself to wait for async tasks (e.g. fluid/terrain generation) to save tokens.",
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {"type": "string", "description": "RPC method to poll, e.g. 'terrain.evaluation_status'"},
                    "params": {"type": "object", "description": "Parameters for the method"},
                    "target_key": {"type": "string", "description": "The JSON key to check in the result, e.g. 'state'"},
                    "target_value": {"type": "string", "description": "The expected value, e.g. 'done'"},
                    "terminal_values": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Values that indicate failure or terminal state, e.g. ['failed', 'cancelled', 'error']. The tool will return immediately if the key matches any of these."
                    },
                    "interval": {"type": "number", "description": "Seconds to wait between checks, e.g. 2.0"},
                    "timeout_sec": {"type": "number", "description": "Maximum total seconds to wait before timing out, e.g. 300.0"}
                },
                "required": ["method", "target_key", "target_value", "interval"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_rpc",
            "description": "Execute one RPC method on the RayTrophi engine.",
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {"type": "string", "description": "Method name, e.g. 'gas.set_settings'"},
                    "params": {"type": "object", "description": "Parameter object for the method"},
                },
                "required": ["method", "params"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "take_screenshot",
            "description": "Capture the current viewport as an image. Use this when you need to visually inspect the scene or verify visual changes. Note: This will consume more tokens.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_script",
            "description": (
                "Write a Python script the engine can run with "
                "execute_rpc('script.run_file', {'path': <returned path>}). Use it "
                "for batch work that would otherwise cost dozens of separate calls "
                "(placing 200 objects, sweeping a parameter). The script runs INSIDE "
                "the engine and uses the rt module (rt.scene, rt.fluid, rt.agent ...), "
                "not this tool list. WARNING: rt module names are NOT the IPC method "
                "names - the IPC method scene.list_objects is rt.scene.objects() in a "
                "script. print() output does not come back, so write results to the "
                "file path the tool returns in 'output_path' and read them with "
                "read_script_output. Returns the absolute path for script.run_file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Bare file name, no directories, e.g. 'scatter_rocks.py'",
                    },
                    "content": {"type": "string", "description": "Full Python source"},
                },
                "required": ["filename", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_script_output",
            "description": (
                "Read a file a script you wrote left behind. This is the only way "
                "to see what a script found: print() from inside the engine does "
                "not travel back over IPC."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string",
                                 "description": "Bare file name, e.g. 'result.json'"},
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delegate_task",
            "description": "Send a prompt/task to another AI agent in the RayTrophi system. Useful for breaking down work or asking an expert agent (like a local code expert) for help.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_agent_id": {"type": "string", "description": "The ID of the target agent, e.g. 'local_qwen' or 'cloud_gemini'"},
                    "task_description": {"type": "string", "description": "The prompt/task to send to the target agent"}
                },
                "required": ["target_agent_id", "task_description"],
            },
        },
    },
]


class ToolExecutor:
    def __init__(self, ipc, registry, report=None):
        self.ipc = ipc
        self.registry = registry
        # Called with (kind, text) so the panel can show what the agent is doing
        # while it is doing it.
        self.report = report or (lambda kind, text: None)

    def get_schemas(self):
        return META_TOOLS_SCHEMA

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _unwrap(response):
        """Return the payload, or a structured error the model can act on."""
        if "error" in response:
            return {"ok": False, "error": response["error"]}
        return response.get("result", response)

    def _short(self, value, limit=280):
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        return text if len(text) <= limit else text[:limit] + " ..."

    # Scripts the model writes land here and nowhere else. The engine is asked
    # to execute whatever path it is handed, so the agent must not be able to
    # aim that at an arbitrary file: a bare name under one directory keeps the
    # blast radius to files this agent itself created.
    SCRIPT_DIR = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "agent_scripts")

    def _write_script(self, filename, content):
        if not filename or not content:
            return {"ok": False, "error": "write_script needs filename and content"}
        base = os.path.basename(filename.strip())
        if base != filename.strip() or os.path.splitext(base)[1] != ".py":
            return {"ok": False,
                    "error": "filename must be a bare .py file name with no "
                             "directory part, e.g. 'scatter_rocks.py'"}
        try:
            os.makedirs(self.SCRIPT_DIR, exist_ok=True)
            path = os.path.join(self.SCRIPT_DIR, base)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(content)
        except OSError as exc:
            return {"ok": False, "error": "could not write script: %s" % exc}
        out_name = os.path.splitext(base)[0] + ".out.json"
        self.report("activity", "wrote script %s (%d lines)"
                    % (base, content.count("\n") + 1))
        return {"ok": True, "path": path,
                "output_path": os.path.join(self.SCRIPT_DIR, out_name),
                "output_filename": out_name,
                "note": "Written, NOT run. Execute it with "
                        "execute_rpc('script.run_file', {'path': '%s'}). A script "
                        "that raises reports the failure there, not here. To see "
                        "what it FOUND, have it write to output_path and then call "
                        "read_script_output('%s')."
                        % (path.replace("\\", "\\\\"), out_name)}

    def _read_script_output(self, filename):
        if not filename:
            return {"ok": False, "error": "read_script_output needs a filename"}
        base = os.path.basename(filename.strip())
        if base != filename.strip():
            return {"ok": False,
                    "error": "filename must be a bare name with no directory part"}
        path = os.path.join(self.SCRIPT_DIR, base)
        if not os.path.exists(path):
            # ★ "The script did not write it" is a different fact from "the script
            # found nothing", and the model must not read one as the other.
            return {"ok": False,
                    "error": "no file named '%s' - the script never wrote it. That "
                             "is not the same as the script finding nothing; check "
                             "that script.run_file actually succeeded." % base}
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                text = handle.read(200000)
        except OSError as exc:
            return {"ok": False, "error": "could not read: %s" % exc}
        self.report("activity", "read script output %s (%d bytes)" % (base, len(text)))
        return {"ok": True, "filename": base, "content": text}

    # -- dispatch -----------------------------------------------------------

    def execute(self, name, args):
        logging.info("tool %s(%s)", name, self._short(args))

        if name == "search_capabilities":
            query = args.get("query", "")
            self.report("activity", "search_capabilities: %s" % query)
            return self._unwrap(self.ipc.call("agent.search_capabilities", {"query": query}))

        if name == "describe_capability":
            method = args.get("method", "")
            self.report("activity", "describe: %s" % method)
            return self.registry.describe_method(self.ipc, method)

        if name == "get_scene_context":
            params = {}
            if "include_probe" in args:
                params["include_probe"] = bool(args["include_probe"])
            self.report("activity", "reading scene context")
            return self._unwrap(self.ipc.call("agent.get_state_summary", params))

        if name == "take_screenshot":
            self.report("activity", "capturing viewport screenshot")
            res_cap = self.ipc.call("viewport.capture", {"enabled": True})
            if "error" in res_cap:
                return self._unwrap(res_cap)
            self.ipc.call("viewport.render_frames", {"count": 1})
            result = self.ipc.call("viewport.get_screenshot", {})
            return self._unwrap(result)

        if name == "write_script":
            return self._write_script(args.get("filename", ""),
                                      args.get("content", ""))

        if name == "read_script_output":
            return self._read_script_output(args.get("filename", ""))

        if name == "delegate_task":
            target = args.get("target_agent_id", "")
            content = args.get("task_description", "")
            self.report("activity", "delegating to %s" % target)
            if not target or not content:
                return {"ok": False,
                        "error": "delegate_task needs both target_agent_id and "
                                 "task_description"}
            # ★ agent.send_prompt, not chat_send. chat_send posts a MESSAGE into
            # the panel; it rejected type "UserPrompt" outright, so delegation
            # failed every time. Only send_prompt reaches the queue another
            # agent polls.
            res_send = self.ipc.call("agent.send_prompt", {
                "sender": "Agent",
                "target": target,
                "content": content,
            })
            result = self._unwrap(res_send)
            if isinstance(result, dict) and result.get("queued"):
                result = dict(result)
                result["note"] = (
                    "Queued only. %s has not run this yet and may never poll - "
                    "do not report the task as done until you have seen its "
                    "result or checked the scene yourself." % target)
            return result

        if name == "execute_rpc":
            method = args.get("method", "")
            params = args.get("params", {}) or {}
            if not isinstance(params, dict):
                return {"ok": False,
                        "error": "params must be an object, got %s" % type(params).__name__}

            # Validate against the real method list, not a prefix. An unknown
            # name is answered with near matches so the model can correct
            # itself instead of guessing again.
            if self.registry.is_bootstrapped and not self.registry.is_method_known(method):
                suggestions = self.registry.suggest(method)
                self.report("error", "unknown method %s" % method)
                return {"ok": False,
                        "error": "no method named '%s' in this build" % method,
                        "did_you_mean": suggestions,
                        "hint": "call search_capabilities or describe_capability first"}

            self.report("activity", "%s %s" % (method, self._short(params, 160)))
            result = self._unwrap(self.ipc.call(method, params))
            # ★★★ This return is load-bearing. Without it the branch fell through
            # to the "unknown tool" line at the bottom, so every engine call was
            # EXECUTED and then reported as a failure the model had to retry -
            # a doubled side effect plus a false failure, which is worse than an
            # error, because the scene really did change.
            if isinstance(result, dict) and result.get("ok") is False:
                self.report("error", "%s: %s"
                            % (method, self._short(result.get("error", ""), 160)))
            return result

        if name == "poll_rpc":
            import time
            method = args.get("method", "")
            params = args.get("params", {}) or {}
            target_key = args.get("target_key", "state")
            target_value = args.get("target_value", "done")
            interval = float(args.get("interval", 2.0))
            timeout_sec = float(args.get("timeout_sec", 300.0))
            terminal_values = args.get("terminal_values", ["failed", "cancelled", "error"])
            if not isinstance(terminal_values, list):
                terminal_values = ["failed", "cancelled", "error"]
            
            self.report("activity", "polling %s for %s=%s" % (method, target_key, target_value))
            
            start_time = time.time()
            while True:
                result = self._unwrap(self.ipc.call(method, params))
                if isinstance(result, dict) and result.get("ok") is False:
                    self.report("error", "%s failed during poll: %s" % (method, result.get("error")))
                    return result
                
                if isinstance(result, dict):
                    current_value = str(result.get(target_key))
                    if current_value == str(target_value):
                        return {"ok": True, "result": result, "note": "Polling completed successfully."}
                    if current_value in terminal_values:
                        self.report("error", "%s entered terminal state: %s" % (method, current_value))
                        return {"ok": False, "error": "Terminal state reached: %s" % current_value, "result": result}
                
                if time.time() - start_time > timeout_sec:
                    return {"ok": False, "error": "Polling timed out after %s seconds." % timeout_sec, "last_result": result}
                
                time.sleep(interval)
                
        return {"ok": False, "error": "unknown tool: %s" % name}
