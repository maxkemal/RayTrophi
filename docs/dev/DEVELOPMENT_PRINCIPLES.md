# RayTrophi Development Principles

> **Durum:** REFERANS — Baglayici muhendislik kurallari.

## Keep foundational files lightweight

Core orchestration headers and source files must remain small enough to review,
compile, and reason about safely.

- Do not keep growing feature catalogs, presets, serializers, UI panels, or
  backend-specific implementations in foundational files.
- When a cohesive block starts making its owner file noticeably larger, split
  it by responsibility at the next safe opportunity.
- Prefer narrowly named modules over generic dumping-ground files.
- Keep the public owner responsible for coordination; move authored data and
  implementation detail into dedicated `.h`, `.cpp`, or `.inl` modules.
- A new feature should not enlarge an already oversized file when a natural
  module boundary exists.
- Refactors must preserve behavior and avoid unrelated cleanup in a dirty
  working tree.

Current application:

- Embedded particle, gas, and fluid production preset implementations live in
  `source/src/Scene/SceneDataParticlePresets.cpp`.
- `scene_data.h` retains only the small public enum, method declaration, and
  SceneData orchestration.
- Prefer a normal `.cpp` translation unit over `.inl` extraction whenever the
  implementation does not require templates or intentional textual inclusion.
