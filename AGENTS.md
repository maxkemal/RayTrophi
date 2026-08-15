# Workspace instructions

- Do not run project builds or application compilation commands (including MSBuild, CMake build, shader compilation, or IDE builds). The user always performs builds.
- Do not launch the RayTrophi Studio application for verification unless the user explicitly asks in that turn.
- Codex may perform read-only inspection, static audits, and non-build source checks, then hand the exact build/test checklist to the user.
- The canonical scene geometry is the flat `TriangleMesh` / DNA SoA representation. New features must read and mutate that flat data path directly; do not use per-face `Triangle` facade collections as scene geometry sources or authoritative state.
- Template, startup, welcome-screen, guided-scene, or related UX work must follow `docs/dev/TEMPLATE_HUB_UX_ROADMAP.md` as the canonical product direction. Preserve the viewport-first UI, reuse the existing left context rail/right contextual dock/bottom editor structure, and avoid introducing a large permanent shelf.
- Do not add feature implementation to an existing source file that is over 2000 lines. Create focused new `.h/.cpp` modules instead; an over-2000-line file may receive only the smallest necessary include, declaration, call, registration, or other integration wiring.
- Every new user-facing system or authoring capability must expose its canonical operations through both the scripting API and IPC. UI, scripting, and IPC must call the same underlying service/core logic; do not create UI-only implementations or duplicate business logic in bindings. A feature is not complete until its script and IPC surfaces, validation/error semantics, and relevant tests/documentation are delivered together.
