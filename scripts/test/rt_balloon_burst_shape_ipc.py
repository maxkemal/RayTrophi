"""Pop shape of the balloon paint demo: uneven ligaments, clumps, blobs, spray.

    python scripts/test/rt_balloon_burst_shape_ipc.py            # Player pipe
    python scripts/test/rt_balloon_burst_shape_ipc.py --studio   # Studio pipe

Needs the demo prepared and autostart stopped (as rt_player_balloon_pop_ipc.py).
Changes no config; pops up to four inflated balloons by id (they come back by
themselves 3.5 s later). Checks, in order:
  1. new binary: status config carries burst_clumpiness / burst_jets_max;
  2. every pop's burst shares add up to its drops (jet + blob + spray);
  3. ligament count within 3..burst_jets_max, clumps present when clumpiness > 0;
  4. uneven: the heaviest ligament carries more than an even share in at least
     one pop (gamma(0.6) shares; an even split would mean the old sphere);
  5. no two pops alike: (jets, blobs, tear) differ between pops.
"""
import json
import sys

from rt_ipc import PLAYER_PIPE_NAME, RtIpc


def main():
    s = RtIpc() if "--studio" in sys.argv else RtIpc(PLAYER_PIPE_NAME)
    report = {}
    state = s.call("paint.wet_demo")
    assert state["prepared"] and not state["running"], "prepare the demo and stop autostart"
    config = state["config"]
    # 1. new binary
    assert "burst_clumpiness" in config and "burst_jets_max" in config, \
        "status config has no burst_clumpiness: old binary"
    clump, jets_max = config["burst_clumpiness"], config["burst_jets_max"]
    report["config"] = {"burst_clumpiness": clump, "burst_jets_max": jets_max}

    targets = [b["id"] for b in state["balloons"] if b["inflated"]][:4]
    assert len(targets) >= 2, "need at least two inflated balloons"
    bursts = []
    for target in targets:
        out = s.call("paint.wet_demo", operation="pop", config={"balloon": target})
        assert out["popped"] == target, out["popped"]
        entry = out["events"][-1]["balloons"][0]
        b = entry["burst"]
        b["drops"] = entry["drops"]
        bursts.append(b)
    report["bursts"] = bursts

    for b in bursts:
        # 2. shares add up
        assert b["jet_drops"] + b["blob_drops"] + b["spray_drops"] == b["drops"], ("shares", b)
        # 3. ligaments / clumps
        if b["jet_drops"] > 0:
            assert 1 <= b["jets"] <= jets_max, ("jets", b)
        if clump > 0.0 and b["jet_drops"] >= 64:
            assert b["clumps"] > 0, ("no clumps", b)
    # 4. uneven
    if clump > 0.0:
        uneven = [b["heaviest_jet_share"] > 1.3 / max(1, b["jets"]) for b in bursts if b["jets"] >= 3]
        report["uneven"] = uneven
        assert any(uneven), ("every ligament split was near even", report)
    # 5. no two alike
    keys = {(b["jets"], b["blobs"], tuple(round(t, 3) for t in b["tear"])) for b in bursts}
    assert len(keys) == len(bursts), ("two pops had the same shape", report)
    print(json.dumps({"result": "PASS", **report}, indent=1))


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(json.dumps({"result": "FAIL", "reason": str(e)}, indent=1, default=str))
        sys.exit(1)
