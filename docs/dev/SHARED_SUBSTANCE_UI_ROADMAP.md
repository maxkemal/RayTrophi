# Shared Substance editor UI follow-up

User direction, 2026-10-18. Recorded for the UI pass; not implemented by the
current acoustics core work. This applies to physical Substance authoring across
the application, not only to audio.

## Layout

Replace the long vertical sequence of physical parameter groups with logical
side-by-side tabs in the existing contextual panel. Start with General,
Thermal/Phase, Fluid, Solid/Grains, Combustion and Acoustics; confirm final grouping
against the canonical SubstanceFieldSpec groups before implementation. Keep
secondary groups inside their relevant tab rather than creating a tab per field
group. Narrow docks need tab overflow/navigation, not clipped labels or a wider
permanent panel. A bounded internal scroll region should keep the editor from
extending the entire inspector/main panel indefinitely. Preserve viewport space
and the existing left rail/right dock/bottom editor structure.

Keep substance identity, built-in/derived status, base profile and assignment
context in a compact shared header. Inheritance/override marks, field units,
validation errors and reset behavior must remain visible and consistent. Add
search across tabs with a clear route to matching fields; unrelated fields should
not disappear permanently merely because a caller is a liquid domain or collider.
Use context to select an appropriate initial tab, while retaining access to the
whole substance. A filter or tab is presentation, not another physics profile.

## One editor for all callers

Evolve the existing `SubstanceEditorUI::draw` / `drawPicker` component in
`source/src/UI/scene_ui_substance_editor.hpp`, not separate copied editors in:

- `ObjectSubstanceUI.cpp`: object physical material / properties inspector;
- `scene_ui_simulation_domains.cpp`: domain defaults and flow-source materials;
- `scene_ui_forcefield.hpp`: collider material and fluid-solver substance pickers;
- any other caller of the shared component, including acoustic material editing.

Use the canonical field registry for ranges, units, groups and validation. All
edits continue through the same Substance service used by Python and IPC. Keep
object assignment, domain default and collider compatibility projections distinct
from editing a shared library profile; show which operation is being performed.
Do not copy substance values into UI-local domain/collider/audio stores.

Per-editor tab/search/scroll state must use stable caller IDs so switching a tab
in one inspector does not change an unrelated editor. Handle in-progress edits
explicitly when switching tabs, assignment, selection or scene: never commit a
draft to a different substance or silently lose a valid edited value. Preserve
the current commit-on-release behavior, avoiding simulation cache invalidation on
every drag pixel. Respect built-in read-only and derive-before-edit semantics.

Prefer units-aware sliders/drag-plus-numeric entry with useful ranges and precise
keyboard editing. Logarithmic controls are appropriate only for positive quantities
with clear zero/disabled semantics; stored physical units and shared validation
must remain unchanged. New authoring operations, if needed, require shared
service/Python/IPC parity; reorganizing tabs alone must not duplicate logic.

## Acceptance for the later UI pass

Check small and wide docks; many fields; tab overflow/search; keyboard navigation;
built-in/derived inheritance; overrides/resets and rejected edits. Open the same
substance through object, domain, flow and collider contexts and verify one edit
is reflected in every view and in script/IPC readback. Verify selection/tab
changes during a drag, undo/redo, scene reload and shared-profile side effects.
Large existing source files receive only integration wiring; place implementation
in focused modules where required by workspace instructions.
