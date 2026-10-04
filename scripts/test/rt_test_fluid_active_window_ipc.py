"""Read-only probe after stepping a localized Vulkan liquid in the application.

python scripts/test/rt_test_fluid_active_window_ipc.py "Grid Domain 1"
Pass --expect-window for a localized, non-periodic liquid acceptance scene.
"""

import argparse
from rt_ipc import RtIpc, RtIpcError


def checked_call(client, method, **params):
    result = client.call(method, **params)
    if result is True:
        return result  # Mutation methods return a boolean success acknowledgement.
    if not isinstance(result, dict):
        raise RtIpcError(f"{method}: expected an object, got {result!r}")
    if result.get("ok") is False:
        raise RtIpcError(f"{method}: {result.get('error', result)}")
    return result


def select_domain(client, requested):
    if requested is not None:
        return requested
    result = checked_call(client, "fluid.list_domains")
    domains = [item for item in result["domains"]
               if item.get("type") in ("fluid", "matter")]
    if len(domains) != 1:
        names = ", ".join(item["name"] for item in domains) or "none"
        raise RtIpcError(f"Specify a liquid domain name; available domains: {names}")
    return domains[0]["name"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain", nargs="?", help="Auto-selects the sole liquid domain if omitted")
    parser.add_argument("--expect-window", action="store_true")
    parser.add_argument("--expect-pressure", action="store_true")
    parser.add_argument("--expect-occupancy", action="store_true")
    args = parser.parse_args()
    client = RtIpc()
    try:
        domain = select_domain(client, args.domain)
        stats = checked_call(client, "fluid.step_stats", domain=domain)
        assert stats.get("measured"), f"Step {domain!r} once before this probe: {stats}"
        required = ("normalize_window_used", "normalize_window_cells", "full_grid_cells")
        missing = [key for key in required if key not in stats]
        assert not missing, f"Running binary lacks active-window telemetry: {missing}"
        used = stats["normalize_window_used"]
        active = stats["normalize_window_cells"]
        full = stats["full_grid_cells"]
        assert full > 0
        if used:
            assert stats["p2g_on_gpu"]
            assert 0 < active < full, (active, full)
        else:
            assert active == 0, "Unused window must not report dispatch savings"
        if args.expect_window:
            assert used, "Localized Vulkan liquid did not use the active normalize path"
        if args.expect_pressure:
            assert stats.get("pressure_window_used"), stats
            assert stats["pressure_on_gpu"], stats
            assert 0 < stats["pressure_window_cells"] < full, stats
        if args.expect_occupancy:
            assert stats.get("occupancy_on_gpu"), stats
        print(f"PASS: {domain}: normalize window used={used}, cells={active}/{full}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
