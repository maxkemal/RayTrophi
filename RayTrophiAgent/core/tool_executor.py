"""The four meta-tools the model is given, and how they reach the engine."""

import json
import logging

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
            if isinstance(result, dict) and result.get("ok") is False:
                self.report("error", "%s failed: %s" % (method, result.get("error")))
            return result

        return {"ok": False, "error": "unknown tool: %s" % name}
