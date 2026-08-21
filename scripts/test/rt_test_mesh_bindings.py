"""In-process Python binding smoke test for the shared mesh preflight API."""

import rt

validation = rt.mesh.validate("Default_Cube")
assert validation["valid"], validation
assert validation["out_of_range_indices"] == 0, validation

preview = rt.mesh.plan_operation(
    "Default_Cube", "edit.extrude", preview=True)
assert preview["ok"], preview
assert preview["backend"] == "auto", preview

blocked = rt.mesh.plan_operation(
    "Default_Cube", "edit.edge_bevel", commit=True)
assert not blocked["ok"], blocked
assert any(d["code"] == "tool_not_executable"
           for d in blocked["diagnostics"]), blocked

positions = rt.mesh.positions("Default_Cube").copy()
positions[0, 0] += 0.125
rt.mesh.set_positions_undoable("Default_Cube", positions)
assert rt.mesh.validate("Default_Cube")["valid"]
