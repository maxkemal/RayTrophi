"""Read live lightning trees after the user build, with timeline Play active.

Checks the actual CPU packet through the existing cloud_stats IPC operation.
Does not alter clouds, play/pause state, or scene assets.
"""
import argparse
import json
import math
import time

from rt_ipc import RtIpc


def grown(progress, birth, completion):
    phase = min(1.0, max(0.0, (progress - birth) / max(completion - birth, 1e-6)))
    t = min(1.0, max(0.0, (phase - 0.35) / 0.55))
    return t * t * (3.0 - 2.0 * t)


def check_tree(event):
    assert event["geometry_revision"] == 2, "Rebuild C++ and all cloud shaders"
    segments, timing = event["segments"], event["segment_growth"]
    assert 32 <= len(segments) <= 80 and len(timing) == len(segments), event
    main = [i for i, metadata in enumerate(timing) if metadata["depth"] == 0]
    assert main == list(range(32)), main
    for i, (segment, metadata) in enumerate(zip(segments, timing)):
        a, b, weight, birth = segment
        completion, parent = metadata["completion"], metadata["parent"]
        assert all(map(math.isfinite, a + b + [weight, birth, completion])), segment
        assert 0 <= birth < completion <= 1.4, metadata
        assert 0 < metadata["tip_weight"] <= weight <= 1, metadata
        assert -1 <= parent < i, metadata
        if parent >= 0:
            assert all(math.isclose(x, y, abs_tol=1e-3) for x, y in
                       zip(a, segments[parent][1])), (i, parent, segment)
            assert birth + 1e-6 >= timing[parent]["completion"], (i, metadata)
        else:
            assert birth == 0, metadata
            assert all(math.isclose(x, y, abs_tol=1e-3) for x, y in
                       zip(a, segments[0][0])), (i, segment)
    assert math.isclose(timing[31]["completion"], 1.0, abs_tol=1e-6)
    for progress in [0, 0.1, 0.3, 0.6, 0.9, 1, 1.2, 1.4]:
        for i, metadata in enumerate(timing):
            parent = metadata["parent"]
            if parent >= 0 and grown(progress, segments[i][3], metadata["completion"]) > 0:
                assert grown(progress, segments[parent][3], timing[parent]["completion"]) >= 0.9999


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=15)
    args = parser.parse_args()
    assert 0 < args.seconds <= 60
    client = RtIpc()
    active = 0
    growing = 0
    branched = 0
    maximum = 0
    deadline = time.monotonic() + args.seconds
    try:
        while time.monotonic() < deadline:
            for backend in client.call("world.cloud_stats")["backends"]:
                for event in backend.get("lightning", []):
                    if not event["active"]:
                        continue
                    check_tree(event)
                    active += 1
                    growing += 0 < event["growth"] < 1
                    branched += len(event["segments"]) > 32
                    maximum = max(maximum, len(event["segments"]))
            time.sleep(0.03)
        assert active, "No live lightning sampled; Play a storm timeline"
        assert growing, "No leader phase sampled; use leader_fraction=0.4..0.6 for this probe"
        assert branched, "No branch sampled; enable branching in the fixture"
        print(json.dumps(dict(passed=True, sampled_events=active, leaders=growing,
                              branched=branched, maximum_segments=maximum)))
    finally:
        client.close()


if __name__ == "__main__":
    main()
