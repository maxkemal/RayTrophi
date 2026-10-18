# Audio gizmo visibility and labels

2026-10-18 source continuation: compact source-only viewport labels now use a
180 px maximum and UTF-8-safe ellipsis. Selected sources take layout priority;
five nearby rows are tried before suppressing a colliding label. Markers remain
available and hover exposes full source name, stable ID and asset name. Existing
occlusion/input gates apply to hover and picking. Names/bindings remain authored
through the same service, Python and IPC operations. Audio source hierarchy is
bounded to eight scrollable rows with search and clipped rendering; the inspector
asset picker also has search. No new script, build or live acceptance was run.
The earlier combined-label description below is superseded by this source change.

2026-10-17 user follow-up: combined source/asset labels are too long in crowded
scenes. The next UI pass should show a short, width-bounded viewport label with
ellipsis; expose full source/asset detail on hover or in the inspector, with
overlap handling. Do not change authored names or IDs to shorten display text.
This is queued work; the combined-label implementation below remains current.
Shared transport scopes and batch processing/A-B controls are tracked in
[the main acoustic roadmap](SCENE_ACOUSTICS_ROADMAP.md).

2026-10-17 source fix; user C++ build and visual acceptance pending.

Audio markers, labels, range rings and directional cones now use the existing
GizmoOcclusion service, its scene BVH, cache, per-frame ray budget and hidden
opacity. Hidden markers do not claim viewport clicks. Selected ImGuizmo transform
handles retain the same always-accessible overlay behavior as other transforms.
BVH occlusion is bypassed during rebuild/animation ownership, as for light probes.

Projection uses the render-image aspect ratio, the shared 0.01 perspective cutoff
and the existing orthographic-height fallback, matching the other scene gizmos.
Labels read the canonical AudioService snapshot: `Audio: source name (asset name)`.
Identical source/asset names are not repeated; missing source names use source ID,
and unbound assets omit the suffix. Rename/relink changes appear next frame.
No source names or asset bindings are modified by this display change.

Existing Python/IPC authoring remains `audio.update(id=..., name=..., asset_id=...)`;
existing viewport gizmo occlusion settings control the shared visibility policy.
No new authoring operation, shader or project registration is required.

User verification after rebuilding Studio:

1. Place two named sources in front of/behind a wall. Confirm the hidden marker,
   text, cone and range-ring sections fade with the other scene gizmos; clicking
   the hidden marker must not intercept selection of the wall in front.
2. Select the hidden source from the source list. Move/rotate using its handles;
   those handles stay accessible. Toggle gizmo occlusion/hidden alpha through the
   existing UI or Python/IPC and confirm audio follows the shared setting.
3. Rename/relink a source; verify both names in the viewport and no duplicate
   suffix when names match. Check unbound sources, perspective/orthographic views,
   nonmatching render/window aspect ratios, and animation/rebuild transitions.

Codex performed source/diff checks only; no Studio launch or compilation.
