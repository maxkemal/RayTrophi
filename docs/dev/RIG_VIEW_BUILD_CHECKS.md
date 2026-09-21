# Skeleton view and selection — user build checks

Status: source implementation and static checks complete; user build/runtime
verification pending. Meshless FBX import itself was confirmed by the user.

## Delivered scope

- X-ray line/joint overlay for visible imported skeletons, including rigs with no mesh.
- Indexed unweighted joints, weighted joints and hierarchy helpers remain inspectable.
- Hierarchy selection and viewport joint picking share `SceneData::rigView`.
- Selected node label/highlight, parent/index/weighted summary and current position.
- A global Skeleton overlay checkbox in each character's existing hierarchy section.
- Evaluated globals captured before skin offsets from Final Pose, legacy controller,
  and ozz LocalToModel output. Per-node bind fallback is reported, not presented as animation.
- Python/IPC share the RigView core through RtApiRig. View selection and visibility
  are transient and not undoable. Reload clears selection and regenerates pose data.

This is the view/selection portion of roadmap Phase 2. Bind editing, bone gizmo
transforms, weight summaries, creation/deletion, independent rig root-motion placement
and retarget authoring remain subsequent work. Phase 0/1/2 as a whole are not complete.

## API contract

Python operations are on `rt.rig`; IPC names carry the `rig.` prefix:

| Operation | Parameters / result |
|---|---|
| `list_characters` | Skeleton-bearing import names; includes rigs without clips |
| `list_bones` | `character` -> array of node data |
| `select_bone` | `character`, `bone` (prefixed node name) |
| `get_selected_bone` | Node data or Python None / JSON null |
| `clear_selection` | Clears bone selection |
| `get_overlay_visible` | Boolean |
| `set_overlay_visible` | `visible` boolean |

Node data: character/name/parent, bone_index (-1 for unindexed helpers), weighted,
pose_source (`bind|graph|controller|ozz`), world_transform (16 row-major floats),
world_position (matrix elements 3, 7, 11). Transform units/axes follow the imported
scene hierarchy. The overlay deliberately draws through surfaces.

Validation errors: unknown_character, character_has_no_skeleton, unknown_bone,
scene_locked during a final render, api_not_bound before binding. Failed selection
preserves prior selection. Removed/reloaded targets return null when selection is
invalid. IPC parameter type failures use invalid_parameter; Python uses normal
binding TypeError. Validated service failures are ValueError in Python and a
machine-readable error code over IPC. Bone names are current canonical references;
future rename/delete services must update selection references atomically.

The first overlay does not implement separate root-motion placement for an unbound
rig. Verify mesh-bound root motion, controller layer blending and nonidentity import
corrections explicitly; source checks are not runtime evidence for those cases.

## Checklist after user build

1. Import `scripts/test/fixtures/rig_authoring/unskinned_three_joint.fbx`. Open
   Characters / Models -> Skeleton. Expect Hip/Knee/Ankle and the Armature ancestor,
   a purple skeleton overlay, zero weighted joints and a bind pose source.
2. Select Knee from the hierarchy. Expect the same node highlighted orange in the
   overlay, its name beside the joint and selection details in the hierarchy panel.
   Click another visible joint in the viewport; the hierarchy selection must follow.
   Repeat with ordinary UI panels hovered and Paint/Sculpt/Edit Mesh active to verify
   skeleton picks do not steal their input. Toggle Skeleton overlay off/on.
3. Import the user's meshless animated FBX. In AnimGraph play Clip -> Final Pose.
   The line/joint overlay must move; pose_source must report graph for evaluated nodes.
   Check paused pose, playback, timeline-follow mode and orthographic/perspective views.
4. Repeat with a mesh-bound character. Weighted joints are cyan. Ensure overlay
   joint positions stay on the character throughout animation, not only in bind pose.
   Repeat with controller playback and ozz where available, multi-character append,
   root motion and weighted controller layers. Check per-node fallback reporting.
5. Run embedded Python contract checks:

   ```python
   import sys
   sys.path.insert(0, "E:/RayTrophi_projesi/raytracing_Proje_Moduler/scripts/test")
   import rig_view_contract
   rig_view_contract.run()
   ```

6. Check animation on separate viewport frames; do not sleep in embedded Python:

   ```python
   character = rt.rig.list_characters()[0]  # Choose the animated rig if several exist.
   before = rt.rig.list_bones(character)
   ```

   Let a moving clip play, then execute:

   ```python
   after = rt.rig.list_bones(character)
   rig_view_contract.assert_pose_changed(before, after)
   ```

7. With the user-started app running, run external IPC contract coverage:

   ```powershell
   python scripts/test/test_rig_view_ipc.py --character YOUR_IMPORT_NAME
   ```

   Selection from IPC must visibly highlight the same node; use rig.select_bone
   followed by viewport capture for this visual parity check. The contract test
   restores prior selection/visibility and does not author geometry or animation.
8. Save/reopen. Skeleton and clips persist; selection starts empty. Both Python
   and IPC list_bones must work again. Run the existing production meshless
   import/save/reopen test on a disposable scene as documented in the fixture README.

Static verification: project XML/source registrations, Python syntax, descriptor
generation and IPC capability/mirror audit passed. No build or app launch was run
by Codex. Runtime tests listed above have not been run in this session.


Segment ownership regression (2026-09-13; after normal user build):
- Humanoid elbow->wrist segment middle selects Forearm (starting joint), not Hand.
  Highlight, elbow gizmo pivot and scalar weight map must agree.
- Wrist joint dot selects Hand even where two segments meet. Hand selection through
  hierarchy/script/IPC highlights wrist->HandEnd; stored skin rows stay unchanged.
- Repeat branching joints, terminal dots, overlapping projected links, scoped Edit
  input locks and Rest/Animated poses. No new test files or Codex app launch.

User confirmed the segment/joint ownership correction works after their build
(2026-09-13). Broader regression cases remain separate acceptance checks.


Phase 2A grouped checks (source delivered, user build/runtime pending):
- Ctrl viewport/tree toggle, Shift full hierarchy range, one active orange joint with
  remaining selected green. Terminal dot/starting-parent segment identity unchanged.
- Owned meshless clip-free Rig Edit: Shift-drag blank space replaces with box selection,
  Ctrl+Shift adds. Escape cancels. Choose Active bone pivot / Selection center.
- Parent+child selected: gizmo Move/Rotate applies once in World/Local modes, unselected
  descendants inherit normal hierarchy motion. G/R changes gizmo mode, S is actor-only.
- Drag Escape preserves original rest/history; release one undo/redo step. Selection,
  pivot, rig revision/load or mode change cancels old preview. Native reopen keeps rest
  and actor but clears transient selection. Weighted/bound rest editing remains gated.
- Optional existing rig_view_contract.py run_multiselection / run_batch_rest on disposable
  owned fixture; IPC --character OwnedRig --multi-selection --batch-rest. Existing native
  roundtrip module run_rest(NEW_OUTPUT.rtp,OwnedRig). No Codex builds or app launch.

### Fit panel coordinate correction (2026-09-13, source delivered)

1. Restore the reported mesh to its original correct viewport orientation; remove
   the compensating 90-degree X rotation. Rebuild normally and Prepare alignment again.
2. Front (X/Y) and Side (Z/Y) mesh samples must match the corresponding viewport views.
3. Move/rotate the mesh and Prepare again, including a multipart target. Bounds and
   samples must follow its placement once; a previous fit preview must reject after changes.
4. Apply manual alignment, preview binding, bind and undo/redo on a disposable owned
   unweighted clip-free rig. Mesh placement must remain stable through binding.

Multi-selection was confirmed by the user. This fit correction has not yet received
user build/runtime verification. Existing C++ regression fixture remains user-run.

### Alignment panel (2026-09-13, source delivered; user build/runtime pending)

1. Rig dock: select a suitable owned meshless clip-free rig and unskinned target,
   Open Alignment. Resize/move the window; change the right-dock context and
   verify the open Alignment window remains visible. Reopen/reload resets drafts.
2. Front and Side: wheel zoom must hold the world point under the cursor; middle
   drag pans only that view. Fit to View / Reset restores all views. Resize columns
   and toggle Top, mesh, skeleton, selected label and opacity. Check smaller windows
   can reach the controls with the scrollbar.
3. Drag a landmark in Front: X/Y changes, Z remains. Side edits Z/Y and preserves X.
   Top edits X/Z and preserves Y. All views and numeric coordinates agree. Use
   the Joint dropdown for overlapping joints. Escape during dragging restores
   the original landmark and does not mutate scene rest or history.
4. Cancel / title-bar close discards the draft. Preparing and navigating must leave
   scene mesh/rig transforms and stored weights unchanged. Enter nonfinite numeric
   coordinates: reject locally without replacing the previous valid landmark.
5. Confirm axes, Preview, Apply. Preview outside-bounds disables Apply. Successful
   Apply gives one production undo/redo step; the window remains with status.
   Move the target or alter rig rest while a draft exists: Preview must require
   Prepare again; modifying a target after Preview must reject stale Apply.
6. Python/IPC get_fit_setup, preview_fit and commit_fit retain the same core
   validation and single history operation used by the panel. Existing runtime
   authoring checks are user-run; Codex does not build or launch the application.

Compare panel Front/Side/Top with viewport Front/Right/Top respectively; signed
Z directions must agree, including after pan/zoom and landmark dragging.

Panel mouse gestures must not start a viewport bone gizmo beneath the window;
while typing in the panel, G/R/S must not change viewport transform mode.

### Apply and bind usability correction

- Confirm axes/rest, edit a valid landmark, then click Apply directly without Preview.
  It must validate and publish one scene-rest undo step. Missing axes confirmation
  must show the reason for disabled Apply.
- Move a landmark outside bounds and click Apply: scene/history must remain unchanged,
  with red joints and clickable outside names. Correct those positions and retry.
- Change target placement/rig after Prepare: direct Apply must require Prepare again.
- After successful Apply, open Bind mesh to skeleton inside Alignment, confirm
  alignment, Preview initial weights, then Bind and apply initial weights. Inspect
  selected-bone maps, deformation and bind undo/redo on a disposable suitable rig.

### Compact Alignment layout

Top uses a two-row compact toolbar; target names and navigation help are in (?).
Joint and XYZ share a row; axes confirmation, Preview/Apply and Close share the
next row. Outside joints use a dropdown. Canvas height reserves measured status
text and the visible footer rows instead of a fixed 245-pixel gap. Check resizing,
long validation messages, outside-joint selection and the post-Apply bind section
after the user's build. Local style must not change other application windows.
Source checks passed; no build or application launch was performed.

### User acceptance update (2026-09-13)

User compiled and confirmed Alignment Apply works, binding works, and initial
weights appear correct. Multi-selection was confirmed earlier. Remaining
checks include bind undo/redo, native reopen and representative posed deformation.
Pose mode, bone Auto Key, IK, footprints and spline locomotion/flight are planned
follow-ups; they are not yet runtime-testable authoring capabilities.

### Phase 2B grouped mirror checks (user-run after normal build)

1. Owned unweighted clip-free humanoid: move LeftUpperArm/LeftForearm, then
   select those sources and Mirror selected. Opposite parent/child must match
   the plane once with valid orientation; one mirror undo/redo restores rest,
   revision and complete selection. Test All Left -> Right and the reverse.
2. Set axis y/z or a nonzero plane offset. Test translated/rotated/scaled actor
   placement; plane remains in rig units. Duplicate/unpaired/both-side selected
   sources reject without scene/history changes. Weighted/bound rigs reject rest
   mirror and creation.
3. Custom owned rig: choose one unpaired non-root bone, set a valid new name and
   Source side, Create opposite bone. Verify shared parent when unpaired and
   mapped opposite parent when paired; new pair visible via get_anatomy. Undo/redo
   must add/remove one node/slot/pair without changing existing indices. Name
   conflict, root and already-paired source must reject.
4. Alignment: Prepare, enable Mirror, drag paired joint in Front/Side/Top and
   edit XYZ; opposite updates using the common plane. Center joints edit alone.
   Escape during drag restores both sides. Offset begins at target bounds center
   in rig space. Directional popup copies all pairs; Cancel changes no scene.
   Apply validates and commits the entire draft once.
5. Save new .rtp/reopen after rest mirror and creation, then inspect hierarchy,
   anatomy pairs and slot identity. No negative scale or duplicate transforms.
6. Optional existing user-run tests: embedded run_mirror("DisposableRig") or
   python scripts/test/test_rig_view_ipc.py --character DisposableRig --mirror.
   Use a fresh owned meshless clip-free fixture; tests explicitly mutate it.

No build, application launch or live tests were run by Codex for this delivery.

### Phase 2B acceptance update (2026-09-13)

User compiled and tested Phase 2B and reported no problems. The delivery is
user-confirmed; this does not claim separate execution of every optional
regression or native round-trip fixture. Next milestone is bound-rig Pose/FK
and bone-channel key authoring/Auto Key (7A/8A).


### Phase 7A/8A Pose/FK and bone keys (user-run after normal build)

Source-delivered; build/runtime checks are pending. Codex has not compiled or
launched the application. This supersedes the earlier statement that Pose/keys
have no source implementation yet; IK and locomotion are still planned.

1. Owned aligned bound rig: select a weighted arm joint and Enter Pose from
   Rig dock. R rotates FK parent and descendants; G translates selected bones.
   Rest hierarchy and weights must remain unchanged. Test Ctrl/Shift multi-select
   parent+child and center/active pivot: no double delta. Scale must be unavailable.
2. Auto Key off: drag, release, Escape during another drag, undo/redo; cancelled
   preview restores the committed pose. Change frame/exit Pose: unkeyed overrides
   clear. Test both GPU and CPU render modes, picking and weight-map alignment.
   Return to Scene with Rest view must restore geometry as well as skeleton.
3. Create clip, frame zero, Key all bones. Go to frame 24, enable Auto Key, rotate
   an arm and release. Preview must deform mesh live; one release writes keys.
   Frame 12 interpolates; frame 24 shows the final pose. Repeated insertion at
   the same frame replaces keys. Undo/redo restores pose/clip keys together.
4. Numeric local position/rotation must preview during editing and commit once
   on edit completion. Escape cancels. Test clip selection, explicit selected
   keys, all keys and the existing timeline scrubbing/playback within Pose.
5. Coverage: Inspect bind coverage / get_pose_coverage reports flat members and
   unweighted count without repairing rows. Intentionally empty rows stay at
   bind position. Observe missing/poor influences under representative poses;
   this phase does not promise anatomically perfect nearest-segment binding.
6. Invalid imported/unbound rigs, duplicate/missing bones, nonfinite or scaled
   matrices, negative/out-of-range frame, stale revision and missing editable
   clip must reject without pose/key mutation. Edit/paint/sculpt mode must not
   steal the Pose gizmo or object selection.
7. Existing optional live fixtures on a DISPOSABLE owned bound rig:
   embedded rig_view_contract.run_pose("DisposableRig") or
   python scripts/test/test_rig_view_ipc.py --character DisposableRig --pose.
   They explicitly create/edit a clip and check preview/cancel, keys, undo/redo,
   independent clip tick rate, rest and weights. Do not run on valuable projects.
8. After keyed changes, use rt_test_rig_roundtrip.run_pose(new_output_path, rig)
   with a NEW .rtp path. Reopen retains authored clip marker, position/quaternion
   keys, rest and weights. Preview/Auto Key/selected clip are session settings;
   the fixture reselects the saved clip and frame explicitly.
9. Existing standalone rig_pose_preview_test.cpp now also links
   RigPoseAuthoringMath.cpp. New cases cover quaternion FK interpolation,
   duplicate-time replacement, rigid/duplicate/unknown validation and atomic
   clip failure. User runs compilation/execution, as with prior fixture changes.

Non-build descriptor audit: PASS, 503 dispatched methods / 34 prefixes,
485 documented / 382 parameterized; security mirror and descriptors current.

Final Pose non-build checks: focused module sizes/includes/delimiters, project
and filter XML/registrations, existing Python regression AST, descriptor JSON
and generated descriptor freshness passed. No build or live test was run.


### Edit/Pose gizmo hover and idle refresh correction (2026-09-13)

User compiled/tested the Pose milestone and confirmed it works, then reported
jitter on mouse hover/click without dragging in Edit/Pose and intermittent drag
startup. Source inspection found ImGuizmo's hover path requests next-frame
WantCaptureMouse, while RigViewportUI treated that same flag as an external UI
capture and skipped Manipulate. This alternated gizmo drawing and blocked clicks.

RigViewportUI now uses real hovered ImGui windows to protect authoring panels;
ImGuizmo's NoInputs overlay and the dock central passthrough do not block the
viewport gizmo. An inherited Scale tool falls back to Translate on Edit/Pose,
so entering these modes cannot silently leave a rigid-only gizmo unavailable.

Pose evaluation now uses transient dirty/frame acknowledgement separate from
the gesture serial. Enter, pose/key/clip mutation, cancel and undo/redo invalidate
evaluation; completed skin evaluation acknowledges it. Frame changes request a
new evaluation. An unchanged preview during a stationary drag is a no-op and
does not reset accumulation. Active-rig clips no longer force idle Pose evaluation;
other characters retain their existing animation evaluation behavior. Shared
UI/Python/IPC pose operations all invalidate via the same API wake path.

Existing rig_pose_preview_test.cpp covers dirty acknowledgement, frame changes
and evaluation invalidation preserving the gesture serial. Its authored-key
fixture now declares its own poseError before use. No build/live tests by Codex;
user rebuild and hover/click/drag verification remain pending for this fix.

User checks after rebuild:
1. Edit and Pose: hover translation arrows/rotation rings for several seconds,
   then click without dragging, then drag. Gizmo must remain visible and responsive.
2. Drag selected parent/child, release, Escape, undo/redo; no dropped gesture.
3. Move mouse into Rig/Alignment panels: viewport gizmo must not steal panel input.
4. Pose idle, stationary mouse-held drag, frame scrub/playback, numeric preview,
   Auto Key and clip selection: redraw only when required; deformation stays current.
5. Enter Edit/Pose after selecting Scale in Scene: Translate is available immediately.


### Gizmo correction user acceptance (2026-09-13)

User rebuilt and tested the correction, confirming that Edit/Pose hover/click
jitter and intermittent manipulation failure are resolved. This supersedes the
pending user verification status above for these reported issues. No separate
execution of every optional regression or native round-trip test is claimed.
