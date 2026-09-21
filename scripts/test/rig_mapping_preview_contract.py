"""Shared embedded Python/external IPC read-only pose-preview contract."""
import math


def check_sample(invoke, params):
    before = invoke("anim.clips", {"character": params["target_character"]})
    first = invoke("anim.sample_clip_binding", dict(params, time_seconds=0.0))
    assert first["binding"]["ready"], first
    assert first["target_pose_source"] == "bound_clip"
    duration = first["duration_seconds"]
    assert math.isfinite(duration) and duration > 0
    sample = invoke("anim.sample_clip_binding", dict(params, time_seconds=duration * 0.25))
    loop = invoke("anim.sample_clip_binding", dict(params, time_seconds=duration))
    for side in ("source", "target"):
        names = {j["name"] for j in sample[side]}
        assert names and len(names) == len(sample[side])
        for joint in sample[side]:
            assert not joint["parent"] or joint["parent"] in names
            m = joint["world_transform"]
            assert len(m) == 16 and all(math.isfinite(x) for x in m)
        assert loop[side] == first[side], "duration boundary must loop to time zero"
    assert invoke("anim.clips", {"character": params["target_character"]}) == before
    assert {m["source"] for m in sample["binding"]["matches"]} <= {j["name"] for j in sample["source"]}
    assert {m["target"] for m in sample["binding"]["matches"]} <= {j["name"] for j in sample["target"]}
    return sample
