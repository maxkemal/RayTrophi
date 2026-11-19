# Player operator console

Source ready, 2026-11-19. C++ build, visual acceptance and physical-camera acceptance
are pending. Codex did not build C++, launch Studio/Player or refresh the distribution.

## Game settings extension / 2026-11-19

Game settings now adds personnel presets, manual/automatic play, balloon count/return,
paint drying/flow/resolution, idle/finale room-reset timers, F8 clear paint, and distinct
session apply versus saved per-game Player defaults. Original projects stay unchanged.
See [PLAYER_GAME_SETTINGS.md](PLAYER_GAME_SETTINGS.md) for the full shared API, local
profile lifecycle, limits and user-build acceptance checklist. Source ready; live
acceptance and updated Player build/package are pending.

## Personnel flow

The temporary console opens over the game viewport at startup. F2 toggles it and
opens Display when reopening; F6 opens Game library. Escape returns to the game.
The original drag/drop project opening remains, using the same scheduled loader as
the library. The console uses the executable's actual RayTrophi icon, native vector
navigation icons, dark cards and mint/violet accents. It is not a permanent shelf.
Existing F1/F3/F4/F5/F9/F11 diagnostics remain; debug camera and balloon clicking
do not consume console interactions. This is the developer operator surface, not
a new PIN/role/security implementation. Existing kiosk key gates still own access.

1. Pick an experience in Game library or drop an `.rtp` file.
2. Choose the screen/projector, render resolution, fullscreen and room volume.
3. If using body input, select the camera, its native capture mode, and Test camera.
4. Mark/save the measured floor rectangle, stop the test, then Start tracking.
5. Check person IDs/confidence in Players, then Play experience.

Play respects the saved balloon demo `autostart` setting: false keeps mouse/body-driven
play and stops any previously enabled random scheduler; true starts the automatic demo.
This rule is shared by the UI, embedded API and IPC `game.start` operator command.
After build: `python scripts/test/rt_player_operator_ipc.py --check-manual-demo`
requires a prepared disposable scene saved with autostart=false.

Stop stops automatic balloon spawning, or uses the configured game session's stop
operation; it is not a global simulation pause. Reset uses the same existing demo
or session reset. For a plain scene, Play returns to the viewer without requiring a demo.

## Game library

Packaged `player_scenes.json` entries and selected library folders are combined.
Add folder opens the native Windows folder picker; manual path entry is also available.
Recursive scanning is asynchronous. It lists `.rtp` and `.RTP` files, removes duplicate
canonical paths, skips `.git`, `.archive`, `node_modules`, `.venv` and directory symlinks.
Limits: 16 roots, depth 6, 50,000 visited entries per root, 4,096 games. Missing folders
and scan limits produce visible messages. Scanning does not load or validate a scene.
Card selection schedules the existing `project.open` service between frames.

Optional artwork: place `name.preview.png` beside `name.rtp`. Images are limited to
16 MB/4096 pixels per dimension on input and reduced to 512 pixels for the bounded
64-image texture cache. Paint/balloon names get balloon fallback artwork; other
projects get a game-controller fallback. No verification screenshot becomes artwork.

Stop operator-owned camera input before switching games. If another API replaces
the scene while input is active, the console requests camera stop and clears body input.

## Camera and calibration

Windows Media Foundation supplies friendly names, symbolic device IDs and the
selected device's native modes. Names with identical models remain distinguishable
by their displayed index and symbolic-link tooltip. Find cameras refreshes the list;
selecting a device refreshes its modes. Mode inspection may activate the chosen
device; actual preview/tracking starts only on an explicit button/action.

Capture runs in a separate hidden, job-owned portable Python process using OpenCV's
MF backend. The worker resolves the selected symbolic link again immediately before
opening its current index. A missing camera fails; camera zero is never a fallback.
Preview and pose tracking have one capture owner. Start/stop and device errors are
shown; stop requests graceful cleanup, with job termination after four seconds if a
driver remains blocked. Process exit clears owned body input and disables its camera
card, including abnormal exits. Preview texture reclamation happens between frames.

The 640x360 RGBA personnel preview is bounded, mutex-protected named shared memory,
updated at up to 15 Hz. Full-resolution processing remains at the selected camera
mode, with processed/requested rates reported separately. Raw images are never saved.
Temporary request/status files contain configuration and aggregate state only; known
session metadata files are removed when the session closes.

Calibration stores the four existing GroundCalibration correspondences and camera
yaw/pitch/roll, using the same file consumed by the bridge. Clicks on a mirrored
personnel preview are converted back to original camera coordinates. The preview
mirror does not mirror the room axes or the game camera card. Changing device or
image dimensions invalidates calibration; frame rate alone does not. Tracking needs
the saved calibration, matching image dimensions and pose model. Camera orientation
and a fifth independent floor point still need on-site measurement/verification.

This slice supports Windows MF colour-camera input. It does not invent a depth stream
or synthetic depth for an RGB camera; existing depth/render paths remain unchanged.
Exposure/focus controls and hardware-specific RGB-D capture are not provided here.

## Camera card and 3D avatars

Camera card uses the existing segmentation/video-card publisher and renderer.
Avatar mode uses `setExternalBodyPose` and the existing canonical flat-mesh rig solver.
It does not introduce another retargeter, skeleton or per-face geometry path.

Players lists scene characters and distinguishes prepared mappings from those needing
mapping. Select a character and a tracked person ID, then Use selected avatar. Existing
external-pose bindings or saved operator profiles supply the mapping. A developer can
import the mapping JSON accepted by `anim.set_external_pose` (anchor bone, controls,
optional blend/confidence/teleport threshold). Missing/invalid mappings report the
existing solver's error rather than pretending the character is ready.

Assignments/profiles are scoped to the canonical scene path; multiple explicitly
assigned characters can bind to different people. Saved assignments are rebound when
tracking starts in avatar mode. Switching to camera card clears external bindings and
hides console-managed avatar meshes. This does not clone arbitrary characters or
automatically assign avatars to every newly detected person. Bindings can be checked
with `anim.external_poses`; real-camera solve/skin behaviour remains a live acceptance item.

## Saved settings and APIs

`player_operator.json` beside the executable stores library roots, selected camera
settings, floor calibration path, display settings, selected screen, master audio and
scene-scoped avatar profiles/assignments. `player_floor_calibration.json` stores floor
correspondences. Packaging copies generated payload without deleting these local files.
Camera/tracking is not automatically started at launch. Explicit CLI display options
take precedence over corresponding saved display fields. Missing screens/devices
are reported. Invalid settings fall back to defaults without rewriting the bad file.
Room audio uses the shared Master mixer; output-device selection remains Windows'
default device, with no fake Player device selector or test-tone implementation.

All console mutations use `rtapi::playerOperator` and the shared service:

```python
# Embedded API registration (requires a Player display host):
rt.player.operator("camera.configure", {"people": 4, "confidence": 0.6})

# Usable in external Player scripts through the existing named-pipe client:
from tools.perception_bridge.operator_client import PlayerOperator
with PlayerOperator() as player:
    state = player.status()
    player.add_folder(r"E:\Games")
    player.command("camera.refresh")
```

The portable package also exposes `from operator_client import PlayerOperator`.
The Player does not gain an embedded Python engine through this feature.
IPC method: `player.operator`, parameters `operation` (default `status`) and `config`
(object). This combined administrative endpoint requires Admin, SceneWrite, FilesRead
and FilesWrite capabilities, including for status. The existing local pipe policy is
unchanged. Use a separate external process, never the embedded script workspace, for
live IPC probes.

| Operation | Config |
|---|---|
| `status` | Empty; returns preferences, games/scan state, camera/catalog, monitors, display, body, characters, audio, error |
| `library.refresh` | Empty |
| `library.add`, `library.remove`, `library.open` | `path` string; open returns scheduled/path |
| `camera.refresh`, `camera.preview`, `camera.start`, `camera.stop` | Empty; start enables tracking, preview does not require model/calibration |
| `camera.configure` | Optional `device_id`, `width`, `height`, `fps`, `people`, `confidence`, `mirror` |
| `calibration.save` | Four `points` with `pixel`/`room_xz` pairs; optional camera yaw/pitch/roll degrees |
| `display.configure` | Existing fullscreen, render_width/render_height, render_scale, pacing_hz, hud |
| `display.monitor` | Zero-based `index` from status.monitors |
| `audio.configure` | Existing Master mixer `gain` and/or `muted` |
| `avatar.bind` | `character`, `person_id`; optional prepared `mapping` |
| `appearance.camera` | Empty |
| `game.start`, `game.stop`, `game.reset` | Empty |

Unknown operations/fields, invalid types/ranges, unavailable devices/modes and edits
during active capture fail explicitly. Camera width/height: 160..8192 / 120..8192;
FPS 1..240; people 1..10; confidence 0.1..1. Actual modes must also be present in the
catalog. Invalid requests do not modify preferences. File publication, rig binding,
window changes or other service failures can leave partial live changes; no transactional
rollback is claimed. Stop is asynchronous. Success is `{ok:true}` unless the operation
returns a specific result. Library and camera inventory are asynchronous status fields.

## Validation and user build checklist

Codex verification:

- 46 perception tests pass, including 11 operator request/lifecycle/preview tests.
- Native Windows MF enumeration rejects a deliberately absent symbolic link without
  opening or falling back to a camera.
- IPC capability mirror/descriptor audit passes; Python syntax and both project
  registrations checked. No C++ build, shader modification, app launch or camera capture.

After incremental Studio and Player builds:

1. Open Player; inspect the panel at 1920x1080, 1280x720 and 800x600, plus a narrow window.
   Check branding, cards, scroll regions, F2/F6/Escape, window/fullscreen and DPI scaling.
2. Add a folder with nested projects, duplicate paths, Turkish filenames, missing assets
   and a `.preview.png`; refresh/remove it. Open a valid card and a dropped project.
3. Run `python scripts/test/rt_player_operator_ipc.py --exercise-library` externally.
   It creates scan fixtures only; it opens no scene or camera and restores its library root.
4. Restart Player: screen/render/audio/camera preferences must persist. Repeat with
   explicit CLI display arguments and a disconnected saved screen.
5. With no camera, Find cameras must show the empty state and game controls remain usable.
6. When cameras arrive, connect two (including duplicate models), select each independently,
   inspect modes and preview. Test busy device, unplug, reconnect/reorder, stop/restart,
   application close and actual FPS. No unexpected device substitution or stale card.
7. Calibrate with measured room orientation; validate a fifth point, mirror preview,
   change dimensions to invalidate, start tracking, check loss/reacquisition and person IDs.
8. Test camera cutout, then a prepared avatar mapping/assignment. Verify live rig/skin,
   multiple explicit assignments, camera-card switch and project replacement cleanup.
9. Repackage using the existing default `scripts/package_player.ps1` only after the build.
   Confirm `perception/bridge/operator_camera.py` and `operator_client.py` are included,
   and local operator/calibration settings remain intact.
