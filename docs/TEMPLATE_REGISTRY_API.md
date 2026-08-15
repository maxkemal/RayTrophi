# Template Registry Script and IPC API

Template Registry is read-only in Phase 1. UI, embedded Python and IPC consume the same `TemplateRegistry` service. Loading or creating projects is intentionally deferred to the transactional loader phase.

## Embedded Python

```python
import rt

rt.templates.refresh()
templates = rt.templates.list()
all_templates = rt.templates.list(include_invalid=True)
metadata = rt.templates.get("raytrophi.start.character_paint")
validation = rt.templates.validate("raytrophi.start.character_paint")
plan = rt.templates.prepare("raytrophi.start.character_paint", conflict_policy="reject")
opened = rt.templates.open("raytrophi.start.empty", conflict_policy="discard")
```

- `refresh()` rescans built-in search roots.
- `list(include_invalid=False)` returns deterministic metadata order.
- `get(id)` returns metadata or raises `KeyError`.
- `validate(id)` returns `id`, `valid` and `errors`, or raises `KeyError`.
- `prepare(id, conflict_policy="reject")` performs a read-only project/recipe preflight and returns a load plan. It never clears or mutates the active project.

## IPC

Read-capability methods:

- `templates.refresh`
- `templates.list` with optional `include_invalid`
- `templates.get` with required string `id`
- `templates.validate` with required string `id`
- `templates.prepare` with required `id` and optional `conflict_policy` (`reject` or `discard`)
- `templates.open` with required `id` and optional `conflict_policy` (`reject` or `discard`)

Example request:

```json
{"id": 1, "method": "templates.list", "params": {"include_invalid": true}}
```

Missing IDs return machine-readable `template_not_found`; invalid parameters return `invalid_parameter`. Registry operations run through the normal main-thread IPC queue.

`prepare` reports `ready`, `state`, `code`, resolved scene/sidecar paths, errors and warnings. With unsaved work, the default `reject` policy produces `unsaved_changes`. The `discard` policy only records explicit intent in the plan during Phase 2A; it does not discard or load anything.

## Phase boundary

Phase 1 registry and Phase 2A preflight do not mutate scene or project state. `templates.open` currently commits only controlled Empty recipes after successful preflight. A rejected conflict or invalid recipe does not mutate the active scene. Script and IPC return the same structured `state`, `code`, `opened`, `ui_state_applied`, `errors`, and `warnings` fields.

Project-backed templates and recipes that require fallible content creation return `transaction_not_available` or `recipe_commit_not_available`. They remain disabled until their content can be staged before the active scene is cleared. The existing `ProjectManager::openProject` is not used by the template commit path.

The Empty recipe contains no geometry or lights, but it creates the temporary viewport camera required by the current editor architecture and marks the editor scene session initialized. This keeps navigation and Solid/Rendered viewport production active without adding an authored scene object.

The General Scene recipe stages a canonical flat `TriangleMesh` cube, camera, and key light before clearing the active project. A staging error returns `recipe_staging_failed` and leaves the active scene untouched. The legacy per-face default-scene creator is not part of this path.
