# Rig import fixtures

`unskinned_three_joint.fbx` is an ASCII FBX 7.4 fixture containing an Armature
transform and three explicit joints: Hip -> Knee -> Ankle. There is no geometry,
skin, blendshape, geometry cache or animation. The translated Armature tests
ancestor transform preservation.

After the user builds, import through the UI or the existing script API:

```python
rt.scene.import_model("E:/RayTrophi_projesi/raytracing_Proje_Moduler/scripts/test/fixtures/rig_authoring/unskinned_three_joint.fbx")
```

Equivalent IPC: `scene.import_model` with `path` set to the fixture's absolute
path. Under Characters / Models -> Skeleton, check the three named joints and
the Armature ancestor. Expect zero weighted bones, zero mesh objects and zero
animation clips. A root helper may also appear; prefixes are generated per import.
Save/reopen and check the hierarchy again. Viewport overlay/picking is a separate
roadmap task; absence of visible mesh pixels is expected for this fixture.

The fixture received static text checks only; it has not been parsed by the
compiled reader or tested in the application in this session.

## Production import and save/reopen regression

The first user runs revealed a remaining final reader gate that required polygon
geometry. That gate now accepts imported joints or clips as well. The collector
unit test alone did not exercise it.

On a disposable scene, run in the embedded Python environment (use a new output
path; saving/opening changes the active project):

```python
import sys
sys.path.insert(0, "E:/RayTrophi_projesi/raytracing_Proje_Moduler/scripts/test")
import rt_test_unskinned_fbx_import
rt_test_unskinned_fbx_import.run("E:/RayTrophi_projesi/raytracing_Proje_Moduler/tmp/unskinned_import_check.rtp")
```

This uses `rt.scene.import_model`, verifies the actual saved joint indices and
parent chain, then reopens and saves to check persistence. Equivalent manual IPC
coverage uses `scene.import_model`, `project.save`, and `project.open` on a new
test project path. For a meshless animated FBX, additionally check
`rt.anim.source_clips()` / `anim.source_clips`, source channels, and playback.
No build or application verification was run by Codex.
