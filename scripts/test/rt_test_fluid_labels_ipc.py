"""Read-only label contract check; run externally after a fresh simulation step.

    python scripts/test/rt_test_fluid_labels_ipc.py "Physics Domain 1"

Pause the timeline first. --allow-unclassified accepts seeded, old disk-cache,
or granular states; it still checks all count invariants. No scene mutation.
"""

import argparse
import math

from rt_ipc import RtIpc, RtIpcError


LABELS = {"unknown", "body", "spray", "foam", "bubble", "mist", "frozen"}


def check_report(info, require_classified):
    report = info["particle_labels"]
    assert report["available"], "no live liquid state"
    assert report["primary_particles"] == info["particle_count"]
    for source in ("primary", "secondary"):
        counts = report[source]
        assert set(counts) == LABELS, (source, counts)
        assert all(type(value) is int and value >= 0 for value in counts.values())
        assert sum(counts.values()) == report[source + "_particles"], source
    assert report["secondary_affects_mass"] is False
    assert report["mist_generation"] is True
    assert math.isclose(report["mist_mass_fraction_max"], 0.15,
                        rel_tol=0.0, abs_tol=1e-6)
    assert report["secondary"]["mist"] == 0
    assert report["render_routing"] == "substance+label"
    assert report["classifier"] == "mass+neighborhood_v2"
    assert report["primary_complete"] == (
        report["primary_particles"] > 0 and report["primary"]["unknown"] == 0
    )
    last = report["last_step"]
    assert type(last["on_gpu"]) is bool
    assert math.isfinite(last["milliseconds"]) and last["milliseconds"] >= 0
    assert 0 <= last["changed"] <= last["particles"]
    assert 0 <= last["center_resolved"] <= last["particles"]
    assert 0 <= last["occupied_bins"] <= last["particles"]
    for key in ("bin_milliseconds", "classify_milliseconds"):
        assert math.isfinite(last[key]) and last[key] >= 0
    assert abs(last["milliseconds"] - last["bin_milliseconds"] -
               last["classify_milliseconds"]) < 0.0001
    if require_classified:
        assert report["primary_complete"], (
            "unclassified or empty state: simulate a fresh non-granular liquid frame; "
            "old disk-cache playback has no labels"
        )
        assert last["particles"] > 0, "classification pass did not run"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain")
    parser.add_argument("--allow-unclassified", action="store_true")
    args = parser.parse_args()
    client = RtIpc()
    try:
        info = client.call("fluid.get", domain=args.domain)
        assert info["type"] == "fluid", "select a liquid domain, not a gas domain"
        report = check_report(info, not args.allow_unclassified)
        listed = client.call("fluid.list_domains")["domains"]
        matches = [item for item in listed if item["name"] == info["name"]]
        assert matches, "domain absent from list_domains"
        for item in matches:
            check_report(item, not args.allow_unclassified)
        print("PASS: get/list expose consistent label count contracts")
        print("primary:", report["primary"])
        print("secondary:", report["secondary"])
        print("classification:", report["last_step"])
        try:
            client.call("fluid.get", domain="__missing_domain_label_contract_94a39d__")
        except RtIpcError:
            print("PASS: missing domain is rejected")
        else:
            raise AssertionError("missing domain was accepted")
    finally:
        client.close()


if __name__ == "__main__":
    main()
