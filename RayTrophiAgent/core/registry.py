"""Local mirror of the engine's method registry.

Bootstraps from agent.discover + agent.list_methods so validation can work on
EXACT method names. The first version only cached domain prefixes, which had
two consequences worth remembering:

  * `scene.does_not_exist` passed validation (harmless - the engine rejects it),
  * and `undo`, `redo`, `batch`, `request_render`, `reset_accumulation` were
    refused forever, because the check required a dot in the name. The agent
    could break things and then could not undo them, and the refusal message
    blamed the registry, teaching the model to correct in the wrong direction.
"""

import logging


class CapabilityRegistry:
    def __init__(self):
        self.domains = {}
        self.methods = {}          # name -> summary/access record
        self.identity = {}
        self.is_bootstrapped = False

    def bootstrap(self, ipc_client):
        logging.info("bootstrapping capability registry from the engine...")
        discovered = ipc_client.call("agent.discover", {})
        if "error" in discovered:
            logging.error("agent.discover failed: %s", discovered["error"])
            return False

        result = discovered.get("result", {})
        self.identity = {
            "app": result.get("app", ""),
            "version": result.get("version", ""),
            "documented_coverage": result.get("documented_coverage"),
            "registered_methods": result.get("registered_methods"),
        }
        for domain in result.get("domains", []):
            self.domains[domain["name"]] = domain

        listed = ipc_client.call("agent.list_methods", {})
        if "error" in listed:
            logging.error("agent.list_methods failed: %s", listed["error"])
            return False
        for entry in listed.get("result", {}).get("methods", []):
            self.methods[entry["method"]] = entry

        self.is_bootstrapped = True
        logging.info("registry ready: %d methods across %d domains (%s %s)",
                     len(self.methods), len(self.domains),
                     self.identity.get("app", "?"), self.identity.get("version", "?"))
        undocumented = [m for m, e in self.methods.items() if not e.get("documented", True)]
        if undocumented:
            logging.warning("%d methods have no written description; their parameter "
                            "schemas are still exact", len(undocumented))
        return True

    # -- validation ---------------------------------------------------------

    def is_method_known(self, method):
        return method in self.methods

    def suggest(self, method, limit=5):
        """Close names, so a rejection can point somewhere instead of just no."""
        if not self.methods:
            return []
        target = method.lower()
        tail = target.split(".")[-1]
        scored = []
        for name in self.methods:
            lowered = name.lower()
            score = 0
            if lowered.split(".")[-1] == tail:
                score += 3
            if tail and tail in lowered:
                score += 2
            if target.split(".")[0] == lowered.split(".")[0]:
                score += 1
            if score:
                scored.append((score, name))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [name for _, name in scored[:limit]]

    def describe_method(self, ipc_client, method):
        if not method:
            return {"error": "no method given"}
        response = ipc_client.call("agent.describe", {"method": method})
        if "error" in response:
            return {"error": response["error"], "did_you_mean": self.suggest(method)}
        return response.get("result", {})
