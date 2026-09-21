"""Shared production Python/IPC contract checks. Requires an imported rig."""
import math


def check(invoke, character=None):
    names = invoke("rig.list_characters", {})
    assert names, "Import the meshless FBX fixture or an animated character first"
    character = character or names[0]
    assert character in names
    bones = invoke("rig.list_bones", {"character": character})
    assert bones
    for bone in bones:
        assert bone["character"] == character
        assert bone["pose_source"] in ("bind", "rest", "graph", "controller", "ozz", "pose", "authored_clip")
        matrix = bone["world_transform"]
        assert len(matrix) == 16 and all(math.isfinite(v) for v in matrix)
        assert bone["world_position"] == [matrix[3], matrix[7], matrix[11]]
    selection = invoke("rig.get_selection", {})
    visible = invoke("rig.get_overlay_visible", {})
    try:
        invoke("rig.select_bone", {"character": character, "bone": bones[-1]["name"]})
        assert invoke("rig.get_selected_bone", {})["name"] == bones[-1]["name"]
        try:
            invoke("rig.select_bone", {"character": character, "bone": "__missing_rig_contract_bone__"})
        except (ValueError, RuntimeError) as error:
            assert "unknown_bone" in str(error)
        else:
            raise AssertionError("Missing bone must be rejected")
        assert invoke("rig.get_selected_bone", {})["name"] == bones[-1]["name"], "Failed selection changed state"
        try:
            invoke("rig.list_bones", {"character": "__missing_rig_contract_character__"})
        except (ValueError, RuntimeError) as error:
            assert "unknown_character" in str(error)
        else:
            raise AssertionError("Missing character must be rejected")
        invoke("rig.clear_selection", {})
        assert invoke("rig.get_selected_bone", {}) is None
        invoke("rig.set_overlay_visible", {"visible": False})
        assert invoke("rig.get_overlay_visible", {}) is False
        invoke("rig.set_overlay_visible", {"visible": True})
        assert invoke("rig.get_overlay_visible", {}) is True
    finally:
        invoke("rig.set_overlay_visible", {"visible": visible})
        if selection["bones"]:
            invoke("rig.select_bones", {"character": selection["character"], "bones": selection["bones"], "active": selection["active"], "anchor": selection["anchor"]})
        else:
            invoke("rig.clear_selection", {})
    print("PASS: rig inventory, finite hierarchy transforms, selection/error semantics, overlay state")
    return bones


def python_invoke(method, params):
    import rt
    return getattr(rt.rig, method.split(".", 1)[1])(**params)


def run(character=None):
    return check(python_invoke, character)


def assert_pose_changed(before, after, tolerance=1e-5):
    """Call after two separate viewport frames while a moving clip plays.

    Do not sleep inside embedded Python: it would block the frame evaluator.
    """
    previous = {bone["name"]: bone for bone in before}
    changed = []
    for bone in after:
        if bone["name"] not in previous or bone["pose_source"] == "bind":
            continue
        if any(abs(a - b) > tolerance for a, b in zip(previous[bone["name"]]["world_transform"], bone["world_transform"])):
            changed.append(bone["name"])
    assert changed, "No evaluated joint transform changed; verify graph playback, source, clip and elapsed frames"
    print("PASS: evaluated hierarchy pose changed for", len(changed), "nodes")
    return changed


def check_weights(invoke, mesh):
    """Shared live Python/IPC read-only weight contract; user supplies a flat mesh.

    Does not import, save, open projects or mutate selection/pose. Call with scene
    animation paused so unrelated frame changes do not affect repeat snapshots.
    """
    selected = invoke("rig.get_selected_bone", {})
    mode = invoke("rig.get_mode", {})
    visible = invoke("rig.get_overlay_visible", {})
    stats = invoke("rig.weight_stats", {"mesh": mesh})
    assert stats["mesh"] == mesh
    assert stats["ownership"] in ("resolved", "unresolved", "ambiguous")
    assert stats["weights_valid"] == (stats["contract_valid"] and stats["fully_weighted"] and stats["bone_indices_verified"])
    count = stats["vertex_count"]
    for vertex in sorted({0, count // 2, count - 1}) if count else []:
        row = invoke("rig.get_weights", {"object": mesh, "vertex": vertex})
        assert row["object"] == mesh and row["vertex"] == vertex and row["vertex_count"] == count
        assert row["ownership"] == stats["ownership"] and row["character"] == stats["character"]
        assert row["influence_count"] == len(row["influences"])
        assert row["unweighted"] == (not row["influences"])
        assert math.isfinite(row["weight_sum"])
        if stats["weights_valid"]:
            assert 1 <= row["influence_count"] <= 4
            assert math.isclose(row["weight_sum"], 1, abs_tol=1e-5)
            assert all(w["value_valid"] and w["index_known"] and w["belongs_to_character"]
                       for w in row["influences"])
        assert invoke("rig.get_weights", {"object": mesh, "vertex": vertex}) == row
    for vertex in (count, 2**64 - 1):
        try:
            invoke("rig.get_weights", {"object": mesh, "vertex": vertex})
        except (ValueError, RuntimeError) as error:
            assert "rig_vertex_out_of_range" in str(error), error
        else:
            raise AssertionError("Out-of-range vertex accepted")
    assert invoke("rig.weight_stats", {"mesh": mesh}) == stats, "Read/error queries changed stored weights"
    assert invoke("rig.get_selected_bone", {}) == selected, "Read/error queries changed bone selection"
    assert invoke("rig.get_mode", {}) == mode, "Read/error queries changed interaction mode"
    assert invoke("rig.get_overlay_visible", {}) == visible, "Read/error queries changed overlay"
    print("PASS: weight identity/coverage, deterministic rows, uint64 bounds and read-only state")
    return stats


def run_weights(mesh):
    return check_weights(python_invoke, mesh)


def check_binding(invoke, character, mesh, undo, redo):
    """Explicit mutating check on a disposable scene; aligned owned meshless rig.

    undo/redo callbacks use the production history. No imports or project reopen.
    """
    from copy import deepcopy
    before = invoke("rig.get_binding", {"character": character})
    assert not before["bound"]
    bones = invoke("rig.list_bones", {"character": character})
    selection = invoke("rig.get_selected_bone", {})
    try:
        invoke("rig.preview_bind", {"character": character, "mesh": mesh})
    except (ValueError, RuntimeError) as error:
        assert "rig_bind_axes_unconfirmed" in str(error), error
    else:
        raise AssertionError("Unconfirmed alignment accepted")
    preview = invoke("rig.preview_bind", {"character": character, "mesh": mesh, "axes_confirmed": True})
    names = [part["mesh"] for part in preview["parts"]]
    stats = {name: invoke("rig.weight_stats", {"mesh": name}) for name in names}
    assert invoke("rig.get_binding", {"character": character}) == before
    assert invoke("rig.list_bones", {"character": character}) == bones
    assert invoke("rig.get_selected_bone", {}) == selection
    stale = deepcopy(preview); stale["mesh_token"] += "stale"
    try:
        invoke("rig.bind_mesh", {"character": character, "mesh": mesh, "preview": stale})
    except (ValueError, RuntimeError) as error:
        assert "rig_bind_stale_preview" in str(error), error
    else:
        raise AssertionError("Stale binding accepted")
    assert invoke("rig.get_binding", {"character": character}) == before
    assert {name: invoke("rig.weight_stats", {"mesh": name}) for name in names} == stats
    forged = deepcopy(preview); forged["can_bind"] = False; forged["parts"] = [{"samples": [{"weights": [[-99, 123]]}]}]
    invoke("rig.bind_mesh", {"character": character, "mesh": mesh, "preview": forged})
    bound = invoke("rig.get_binding", {"character": character})
    assert bound["bound"] and [part["mesh"] for part in bound["parts"]] == names
    assert all(part["present"] for part in bound["parts"])
    for name in names:
        current = check_weights(invoke, name)
        assert current["weights_valid"] and current["character"] == character
    undo()
    assert invoke("rig.get_binding", {"character": character}) == before
    assert {name: invoke("rig.weight_stats", {"mesh": name}) for name in names} == stats
    redo()
    assert invoke("rig.get_binding", {"character": character}) == bound
    for name in names:
        assert invoke("rig.weight_stats", {"mesh": name})["weights_valid"]
    invoke("rig.unbind_mesh", {"character": character})
    unbound = invoke("rig.get_binding", {"character": character})
    assert not unbound["bound"] and not unbound["parts"]
    assert invoke("rig.list_bones", {"character": character}), "Unbind removed skeleton"
    invoke("rig.set_mode", {"mode": "pose", "character": character})
    assert invoke("rig.get_mode", {})["mode"] == "pose"
    invoke("rig.set_mode", {"mode": "scene", "character": character})
    undo()
    assert invoke("rig.get_binding", {"character": character}) == bound
    redo()
    assert not invoke("rig.get_binding", {"character": character})["bound"]
    undo()  # Leave the disposable fixture bound for the remaining weight tests.
    assert invoke("rig.get_binding", {"character": character}) == bound
    print("PASS: bind/unbind geometry ownership, computed weights and undo/redo")
    return bound


def run_binding(character, mesh):
    import rt
    return check_binding(python_invoke, character, mesh, rt.undo, rt.redo)


def check_weight_map(invoke, mesh, character, bone):
    """Shared Python/IPC display-field checks; paused weighted flat fixture."""
    selected = invoke("rig.get_selected_bone", {})
    mode = invoke("rig.get_mode", {})
    stats = invoke("rig.weight_stats", {"mesh": mesh})
    visible = invoke("rig.get_weight_map_visible", {})
    try:
        for value in (True, False):
            invoke("rig.set_weight_map_visible", {"visible": value})
            assert invoke("rig.get_weight_map_visible", {}) == value
        field = invoke("rig.get_weight_map", {"mesh": mesh, "character": character, "bone": bone})
        assert field["vertex_count"] == stats["vertex_count"] == len(field["values"])
        assert field["character"] == character and field["bone"] == bone
        assert all(math.isfinite(value) and 0 <= value <= 1 for value in field["values"])
        for vertex in sorted({0, len(field["values"]) // 2, len(field["values"]) - 1}) if field["values"] else []:
            row = invoke("rig.get_weights", {"object": mesh, "vertex": vertex})
            expected = sum(w["weight"] for w in row["influences"] if w["bone_index"] == field["bone_index"]
                           and w["weight"] is not None and w["weight"] >= 0)
            assert math.isclose(field["values"][vertex], min(1, max(0, expected)), abs_tol=1e-6)
        assert invoke("rig.get_weight_map", {"mesh": mesh, "character": character, "bone": bone}) == field
        for params, code in (({"mesh": mesh, "character": character, "bone": "__missing_bone__"}, "unknown_bone"),
                             ({"mesh": mesh, "character": "__foreign_character__", "bone": bone}, "rig_weight_map_foreign_character")):
            try:
                invoke("rig.get_weight_map", params)
            except (ValueError, RuntimeError) as error:
                assert code in str(error), error
            else:
                raise AssertionError("Invalid weight map request accepted")
        assert invoke("rig.weight_stats", {"mesh": mesh}) == stats
        assert invoke("rig.get_selected_bone", {}) == selected and invoke("rig.get_mode", {}) == mode
        print("PASS: canonical scalar rows, clamp, owner/errors and independent display toggle")
        return field
    finally:
        invoke("rig.set_weight_map_visible", {"visible": visible})


def run_weight_map(mesh, character, bone):
    return check_weight_map(python_invoke, mesh, character, bone)


def restore_selection(invoke, selection):
    if selection["bones"]:
        invoke("rig.select_bones", {"character": selection["character"], "bones": selection["bones"], "active": selection["active"], "anchor": selection["anchor"]})
    else:
        invoke("rig.clear_selection", {})
    invoke("rig.set_selection_pivot", {"mode": selection["pivot"]})


def check_multiselection(invoke, character):
    """Shared transient selection contract; >=3 joints, no scene geometry edits."""
    before = invoke("rig.get_selection", {})
    views = invoke("rig.list_bones", {"character": character})
    names = [b["name"] for b in views]
    assert len(names) >= 3
    try:
        invoke("rig.select_bones", {"character": character, "bones": names[:2], "active": names[0]})
        assert set(invoke("rig.get_selection", {})["bones"]) == set(names[:2])
        assert invoke("rig.get_selected_bone", {})["name"] == names[0]
        invoke("rig.select_bones", {"character": character, "bones": [names[-1]], "mode": "add"})
        assert invoke("rig.get_selection", {})["active"] == names[-1]
        invoke("rig.select_bones", {"character": character, "bones": [names[-1]], "mode": "toggle"})
        snap = invoke("rig.get_selection", {})
        assert set(snap["bones"]) == set(names[:2]) and snap["active"] in names[:2]
        invalid = ((dict(bones=[names[0], names[0]]), "rig_selection_duplicate_bone"),
                   (dict(bones=["__missing_bone__"]), "unknown_bone"),
                   (dict(bones=[names[0]], active=names[-1]), "rig_selection_active_not_selected"),
                   (dict(bones=[names[0]], anchor=names[-1]), "rig_selection_anchor_not_selected"),
                   (dict(bones=[names[0]], mode="invalid"), "rig_selection_invalid_mode"))
        for params, code in invalid:
            try:
                invoke("rig.select_bones", dict(character=character, **params))
            except (ValueError, RuntimeError) as error:
                assert code in str(error), error
            else:
                raise AssertionError("Invalid selection accepted")
            assert invoke("rig.get_selection", {}) == snap
        invoke("rig.select_bone", {"character": character, "bone": names[0]})
        assert invoke("rig.get_selection", {})["bones"] == [names[0]]
        invoke("rig.select_bones", {"character": character, "bones": [names[-1]], "mode": "range"})
        children = {}
        for b in views:
            children.setdefault(b["parent"], []).append(b["name"])
        order = []
        def visit(name):
            order.append(name)
            for child in children.get(name, []):
                visit(child)
        for b in views:
            if not b["parent"] or b["parent"] not in names:
                visit(b["name"])
        a, b = sorted((order.index(names[0]), order.index(names[-1])))
        selection = invoke("rig.get_selection", {})
        assert set(selection["bones"]) == set(order[a:b + 1]) and selection["active"] == names[-1]
        assert selection["anchor"] == names[0]
        restore_selection(invoke, selection)
        assert invoke("rig.get_selection", {}) == selection
        for pivot in ("center", "active"):
            invoke("rig.set_selection_pivot", {"mode": pivot})
            assert invoke("rig.get_selection", {})["pivot"] == pivot
        assert invoke("rig.list_bones", {"character": character}) == views
        print("PASS: replace/add/toggle/range, active identity, legacy select, errors and transient pivot")
    finally:
        restore_selection(invoke, before)


def check_batch_rest(invoke, character, undo, redo):
    """Explicit mutating check on a disposable owned meshless clip-free fixture."""
    views = invoke("rig.list_bones", {"character": character})
    parent = next((p for p in views if any(c["parent"] == p["name"] for c in views)), None)
    assert parent and parent["authoring_owned"] and not any(b["weighted"] for b in views)
    child = next(b for b in views if b["parent"] == parent["name"])
    names = [parent["name"], child["name"]]
    invoke("rig.select_bones", {"character": character, "bones": names, "active": child["name"]})
    selection = invoke("rig.get_selection", {})
    delta = [1,0,0,.1, 0,1,0,.05, 0,0,1,0, 0,0,0,1]
    invoke("rig.transform_rest", {"character": character, "bones": names, "world_delta": delta, "rig_revision": parent["rig_revision"]})
    after = invoke("rig.list_bones", {"character": character})
    by_name = {b["name"]: b for b in after}
    for before in (parent, child):
        current = by_name[before["name"]]
        assert current["bone_index"] == before["bone_index"]
        for old, new, amount in zip(before["world_position"], current["world_position"], (.1, .05, 0)):
            assert math.isclose(new - old, amount, abs_tol=1e-5), "Selected parent+child moved twice"
        assert all(current[key] for key in ("in_bonedata", "in_skeleton_nodes", "in_node_hierarchy", "in_ozz_skeleton"))
    try:
        invoke("rig.transform_rest", {"character": character, "bones": names, "world_delta": delta, "rig_revision": parent["rig_revision"]})
    except (ValueError, RuntimeError) as error:
        assert "rig_edit_stale_revision" in str(error), error
    else:
        raise AssertionError("Stale edit accepted")
    assert invoke("rig.list_bones", {"character": character}) == after
    undo()
    assert invoke("rig.list_bones", {"character": character}) == views
    assert invoke("rig.get_selection", {}) == selection
    redo()
    assert invoke("rig.list_bones", {"character": character}) == after
    assert invoke("rig.get_selection", {}) == selection
    print("PASS: batch world delta once, representations/stable indices, stale rejection and one-step undo/redo")


def run_multiselection(character):
    return check_multiselection(python_invoke, character)


def run_batch_rest(character):
    import rt
    return check_batch_rest(python_invoke, character, rt.undo, rt.redo)


def check_mirror(invoke, character, undo, redo):
    """Explicit mutating test on disposable owned meshless clip-free fixture.

    Leaves source edit and mirror committed. If no pairs exist, creates an
    opposite non-root bone and exercises production creation undo/redo first.
    """
    views = invoke("rig.list_bones", {"character": character})
    anatomy = invoke("rig.get_anatomy", {"character": character})
    selection = invoke("rig.get_selection", {})
    if not anatomy["symmetry"]:
        source = next(b for b in views if b["parent"])
        name = invoke("rig.get_next_bone_name", {"character": character, "seed": "MirrorJoint"})
        invoke("rig.create_mirrored_bone", dict(character=character, bone=source["name"], name=name,
               source_side="left", rig_revision=source["rig_revision"]))
        created = invoke("rig.list_bones", {"character": character})
        created_anatomy = invoke("rig.get_anatomy", {"character": character})
        assert len(created) == len(views) + 1
        assert created_anatomy["symmetry"][0]["left"] == source["name"]
        assert all(b[key] for b in created for key in ("in_bonedata", "in_skeleton_nodes", "in_node_hierarchy", "in_ozz_skeleton"))
        undo()
        assert invoke("rig.list_bones", {"character": character}) == views
        assert invoke("rig.get_anatomy", {"character": character}) == anatomy
        assert invoke("rig.get_selection", {}) == selection
        redo()
        assert invoke("rig.list_bones", {"character": character}) == created
        assert invoke("rig.get_anatomy", {"character": character}) == created_anatomy
        views, anatomy = created, created_anatomy
    pair = anatomy["symmetry"][0]
    source = pair["left"]
    delta = [1,0,0,0, 0,1,0,.137, 0,0,1,0, 0,0,0,1]
    invoke("rig.transform_rest", dict(character=character, bones=[source], world_delta=delta, rig_revision=views[0]["rig_revision"]))
    before = invoke("rig.list_bones", {"character": character})
    old_selection = invoke("rig.get_selection", {})
    marks = {b["name"]: b["world_position"] for b in before}
    mirrored = invoke("rig.get_mirrored_landmarks", dict(character=character, landmarks=marks,
                      bones=[source], rig_revision=before[0]["rig_revision"]))
    assert invoke("rig.list_bones", {"character": character}) == before
    assert invoke("rig.get_selection", {}) == old_selection
    assert marks[source] == mirrored[source]
    invoke("rig.mirror_rest", dict(character=character, bones=[source], rig_revision=before[0]["rig_revision"]))
    after = invoke("rig.list_bones", {"character": character})
    target = next(b for b in after if b["name"] == pair["right"])
    assert all(math.isclose(x,y,abs_tol=1e-4) for x,y in zip(target["world_position"], mirrored[pair["right"]]))
    assert {b["name"]: b["bone_index"] for b in after} == {b["name"]: b["bone_index"] for b in before}
    assert invoke("rig.get_selection", {})["bones"] == [pair["right"]]
    try:
        invoke("rig.mirror_rest", dict(character=character,bones=[source],rig_revision=before[0]["rig_revision"]))
    except (ValueError, RuntimeError) as error:
        assert "rig_edit_stale_revision" in str(error), error
    else:
        raise AssertionError("Stale mirror accepted")
    assert invoke("rig.list_bones", {"character": character}) == after
    undo()
    assert invoke("rig.list_bones", {"character": character}) == before
    assert invoke("rig.get_selection", {}) == old_selection
    redo()
    assert invoke("rig.list_bones", {"character": character}) == after
    print("PASS: mirror creation/pairs, read-only landmarks, stable indices, stale rejection and undo/redo")


def run_mirror(character):
    import rt
    return check_mirror(python_invoke, character, rt.undo, rt.redo)


def check_pose(invoke, character, undo, redo):
    """Mutates a disposable owned bound rig; no viewport waits inside Python."""
    import time
    invoke("rig.set_mode", {"mode": "scene"})
    before = invoke("rig.list_bones", {"character": character})
    assert before and all(b["authoring_owned"] for b in before)
    coverage = invoke("rig.get_pose_coverage", {"character": character})
    assert coverage["vertex_count"] > 0 and coverage["unweighted_behavior"] == "bind_position"
    meshes = [part["mesh"] for part in coverage["parts"]]
    weights = {mesh: invoke("rig.get_weights", {"object": mesh, "vertex": 0}) for mesh in meshes}
    rest = {b["name"]: b["local_rest_transform"] for b in before}
    invoke("rig.set_mode", {"mode": "pose", "character": character})
    assert invoke("rig.get_mode", {})["mode"] == "pose"
    clip = "PoseContract_" + str(time.time_ns())
    invoke("rig.create_pose_clip", {"character": character, "name": clip, "fps": 30})
    invoke("rig.set_pose_frame", {"frame": 24})
    state = invoke("rig.get_pose_state", {"character": character})
    bone = next(b["name"] for b in before if b["weighted"])
    transform = list(state["local_transforms"][bone])
    transform[3] += .05
    params = dict(character=character, local_transforms={bone: transform}, rig_revision=state["rig_revision"])
    invoke("rig.preview_pose_locals", params)
    assert invoke("rig.get_pose_state", {"character": character})["preview"]
    invoke("rig.cancel_pose_preview", {"character": character})
    assert invoke("rig.get_pose_state", {"character": character})["local_transforms"] == state["local_transforms"]
    invoke("rig.set_pose_auto_key", {"enabled": True})
    invoke("rig.preview_pose_locals", params)
    invoke("rig.apply_pose_preview", {"character": character})
    keyed = invoke("rig.get_pose_state", {"character": character})
    keys = lambda data: next(c["rotation_keys"] for c in data["clips"] if c["name"] == clip)
    assert keys(keyed) == keys(state) + 1 and not keyed["preview"]
    undo()
    assert keys(invoke("rig.get_pose_state", {"character": character})) == keys(state)
    redo()
    assert keys(invoke("rig.get_pose_state", {"character": character})) == keys(keyed)
    invoke("rig.set_pose_frame", {"frame": 0})
    zero = invoke("rig.get_pose_state", {"character": character})["local_transforms"][bone][3]
    invoke("rig.set_pose_frame", {"frame": 12})
    middle = invoke("rig.get_pose_state", {"character": character})["local_transforms"][bone][3]
    invoke("rig.set_pose_frame", {"frame": 24})
    last = invoke("rig.get_pose_state", {"character": character})["local_transforms"][bone][3]
    assert math.isclose(middle, (zero + last) / 2, abs_tol=1e-5)
    assert math.isclose(last, transform[3], abs_tol=1e-5)
    invalid = dict(params, rig_revision=state["rig_revision"] + 1)
    try:
        invoke("rig.preview_pose_locals", invalid)
    except (ValueError, RuntimeError) as error:
        assert "rig_edit_stale_revision" in str(error), error
    else:
        raise AssertionError("Stale pose revision accepted")
    assert not invoke("rig.get_pose_state", {"character": character})["preview"]
    assert {b["name"]: b["local_rest_transform"] for b in invoke("rig.list_bones", {"character": character})} == rest
    assert invoke("rig.get_pose_coverage", {"character": character}) == coverage
    assert {mesh: invoke("rig.get_weights", {"object": mesh, "vertex": 0}) for mesh in meshes} == weights
    invoke("rig.set_pose_auto_key", {"enabled": False})
    invoke("rig.set_mode", {"mode": "scene"})
    print("PASS: bound Pose preview/cancel, native keys, Auto Key, undo/redo, FPS interpolation and unchanged rest/weights")


def run_pose(character):
    import rt
    return check_pose(python_invoke, character, rt.undo, rt.redo)


def check_pose_mirror(invoke, character, undo, redo):
    """Use a disposable owned bound rig with anatomy pairs; creates a native clip."""
    import time
    params = {"character": character}
    invoke("rig.set_mode", {"mode": "scene"})
    bones = invoke("rig.list_bones", params)
    rest = {b["name"]: b["local_rest_transform"] for b in bones}
    anatomy = invoke("rig.get_anatomy", params)
    pair = anatomy["symmetry"][0]
    source, target = pair["left"], pair["right"]
    coverage = invoke("rig.get_pose_coverage", params)
    weights = {part["mesh"]: invoke("rig.get_weights", {"object": part["mesh"], "vertex": 0}) for part in coverage["parts"]}
    invoke("rig.set_mode", dict(params, mode="pose"))
    invoke("rig.create_pose_clip", dict(params, name="MirrorContract_" + str(time.time_ns()), fps=24))
    invoke("rig.set_pose_frame", {"frame": 24})
    invoke("rig.set_pose_auto_key", {"enabled": True})
    state = invoke("rig.get_pose_state", params)
    # Rotate around source local Z without changing the rest translation.
    local = state["local_transforms"][source]
    angle = .15
    cs, sn = math.cos(angle), math.sin(angle)
    changed = list(local)
    for r in range(3):
        changed[r*4] = local[r*4]*cs + local[r*4+1]*sn
        changed[r*4+1] = -local[r*4]*sn + local[r*4+1]*cs
    invoke("rig.preview_pose_locals", dict(params, local_transforms={source: changed}, rig_revision=state["rig_revision"]))
    invoke("rig.apply_pose_preview", params)
    before = invoke("rig.get_pose_state", params)
    mirror = dict(params, bones=[source], rig_revision=state["rig_revision"])
    invoke("rig.mirror_pose", mirror)
    preview = invoke("rig.get_pose_state", params)
    assert preview["preview"] and preview["local_transforms"][source] == before["local_transforms"][source]
    invoke("rig.cancel_pose_preview", params)
    assert invoke("rig.get_pose_state", params)["local_transforms"] == before["local_transforms"]
    for invalid, expected in ((dict(mirror, bones=[source,target]), "rig_mirror_ambiguous_pair"),
                              (dict(mirror, rig_revision=state["rig_revision"]+1), "rig_edit_stale_revision")):
        try:
            invoke("rig.mirror_pose", invalid)
        except (ValueError, RuntimeError) as error:
            assert expected in str(error), error
        else:
            raise AssertionError("Invalid pose mirror accepted")
    invoke("rig.set_pose_auto_key", {"enabled": True})
    invoke("rig.mirror_pose", mirror)
    invoke("rig.apply_pose_preview", params)
    after = invoke("rig.get_pose_state", params)
    assert not after["preview"]
    assert any(not math.isclose(x,y,abs_tol=1e-6) for x,y in zip(before["local_transforms"][target], after["local_transforms"][target])), "Use an unrestricted pair for this fixture"
    undo()
    assert invoke("rig.get_pose_state", params)["local_transforms"] == before["local_transforms"]
    redo()
    assert invoke("rig.get_pose_state", params)["local_transforms"] == after["local_transforms"]
    invoke("rig.set_pose_frame", {"frame": 0})
    invoke("rig.set_pose_frame", {"frame": 24})
    sampled = invoke("rig.get_pose_state", params)["local_transforms"][target]
    assert all(math.isclose(x,y,abs_tol=1e-5) for x,y in zip(sampled,after["local_transforms"][target]))
    assert {b["name"]: b["local_rest_transform"] for b in invoke("rig.list_bones", params)} == rest
    assert invoke("rig.get_pose_coverage", params) == coverage
    assert {mesh: invoke("rig.get_weights", {"object": mesh, "vertex": 0}) for mesh in weights} == weights
    invoke("rig.set_pose_auto_key", {"enabled": False})
    invoke("rig.set_mode", {"mode": "scene"})
    print("PASS: pose mirror preview/cancel, pair/revision errors, Auto Key, undo/redo and native channel sampling")


def run_pose_mirror(character):
    import rt
    return check_pose_mirror(python_invoke, character, rt.undo, rt.redo)


def check_joint_profile(invoke, character, undo, redo):
    """Mutates a disposable owned bound rig, then restores its original profile."""
    invoke("rig.set_mode", {"mode": "scene"})
    params = {"character": character}
    original = invoke("rig.get_joint_profile", params)
    coverage = invoke("rig.get_pose_coverage", params)
    bones = invoke("rig.list_bones", params)
    rest = {b["name"]: b["local_rest_transform"] for b in bones}
    bone = next(b["name"] for b in bones if b["weighted"] and b["parent"])
    proposal = invoke("rig.suggest_joint_profile", params)
    assert proposal["needs_review"] and all(not row["enabled"] for row in proposal["profile"]["joints"])
    assert invoke("rig.get_joint_profile", params) == original
    profile = {"version": 1, "joints": [dict(row) for row in original["profile"]["joints"] if row["bone"] != bone]}
    row = dict(bone=bone, type="fixed", enabled=True, lock_translation=True,
               axis=[1, 0, 0], minimum=-180, maximum=180, swing=180)
    profile["joints"].append(row)
    invoke("rig.set_joint_profile", dict(character=character, profile=profile, rig_revision=original["rig_revision"]))
    changed = invoke("rig.get_joint_profile", params)
    assert changed["profile"] == profile and changed["rig_revision"] == original["rig_revision"] + 1
    invalid = {"version": 1, "joints": [dict(row, axis=[0, 0, 0])]}
    for data, revision, expected in [(invalid, changed["rig_revision"], "rig_joint_invalid_axis"),
                                     (original["profile"], original["rig_revision"], "rig_edit_stale_revision")]:
        try:
            invoke("rig.set_joint_profile", dict(character=character, profile=data, rig_revision=revision))
        except (ValueError, RuntimeError) as error:
            assert expected in str(error), error
        else:
            raise AssertionError("Invalid/stale joint profile accepted")
    assert invoke("rig.get_joint_profile", params) == changed
    undo()
    assert invoke("rig.get_joint_profile", params) == original
    redo()
    assert invoke("rig.get_joint_profile", params) == changed
    invoke("rig.set_mode", {"mode": "pose", "character": character})
    invoke("rig.set_pose_frame", {"frame": 0})
    before = invoke("rig.get_pose_state", params)
    matrix = list(before["local_transforms"][bone])
    matrix[3] += .1
    invoke("rig.preview_pose_locals", dict(character=character, local_transforms={bone: matrix}, rig_revision=changed["rig_revision"]))
    limited = invoke("rig.get_pose_state", params)
    assert bone in limited["limit_hits"]
    assert all(math.isclose(a,b,abs_tol=1e-5) for a,b in zip(limited["local_transforms"][bone],before["local_transforms"][bone]))
    invoke("rig.apply_pose_preview", params)  # A blocked request is a successful no-op.
    after = invoke("rig.get_pose_state", params)
    assert not after["preview"] and after["clips"] == before["clips"]
    assert {b["name"]: b["local_rest_transform"] for b in invoke("rig.list_bones", params)} == rest
    assert invoke("rig.get_pose_coverage", params) == coverage
    invoke("rig.set_mode", {"mode": "scene"})
    invoke("rig.set_joint_profile", dict(character=character, profile=original["profile"], rig_revision=changed["rig_revision"]))
    assert invoke("rig.get_joint_profile", params)["profile"] == original["profile"]
    print("PASS: joint proposal/read, profile validation/history, constrained shared Pose, no-op keys and unchanged rest/weights")


def run_joint_profile(character):
    import rt
    return check_joint_profile(python_invoke, character, rt.undo, rt.redo)
