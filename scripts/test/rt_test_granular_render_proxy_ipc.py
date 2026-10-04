"""IPC contract test for render-only fluid particle refinement.

Run from a separate terminal while RayTrophi Studio is open:

    python scripts/test/rt_test_granular_render_proxy_ipc.py "Grid Domain 1"

The test changes only render settings and restores them on exit.
"""

import sys

from rt_ipc import RtIpc, RtIpcError


DOMAIN = sys.argv[1] if len(sys.argv) > 1 else "Grid Domain 1"


def expect_refused(client, payload, label):
    try:
        client.call("fluid.set_splat_geometry", **payload)
    except RtIpcError:
        print("[PASS] " + label)
        return
    raise AssertionError(label + " was accepted")


def listed_domain(client):
    domains = client.call("fluid.list_domains")["domains"]
    return next(item for item in domains if item["name"] == DOMAIN)


def main():
    client = RtIpc()
    try:
        before = client.call("fluid.get", domain=DOMAIN)
        restore = {
            "domain": DOMAIN,
            "geometry": before["splat_geometry"],
            "virtual_grains": before["virtual_grains_requested"],
            "grain_size_variation": before["grain_size_variation"],
            "virtual_grain_budget": before["virtual_grain_budget"],
            "granular_physical_carriers":
                before["granular_physical_carriers"],
        }
        try:
            expect_refused(
                client,
                {"domain": DOMAIN, "virtual_grains": 0},
                "zero virtual grains rejected")
            expect_refused(
                client,
                {"domain": DOMAIN, "virtual_grains": 33},
                "more than 32 virtual grains rejected")
            expect_refused(
                client,
                {"domain": DOMAIN, "grain_size_variation": -0.01},
                "negative size variation rejected")
            expect_refused(
                client,
                {"domain": DOMAIN, "grain_size_variation": 0.76},
                "size variation above 0.75 rejected")
            expect_refused(
                client,
                {"domain": DOMAIN, "virtual_grain_budget": 999999},
                "budget below one million rejected")
            expect_refused(
                client,
                {"domain": DOMAIN, "virtual_grain_budget": 32000001},
                "budget above 32 million rejected")

            client.call(
                "fluid.set_splat_geometry",
                domain=DOMAIN,
                geometry="icosphere",
                virtual_grains=7,
                grain_size_variation=0.35,
                virtual_grain_budget=12000000,
                granular_physical_carriers=False)
            for info in (client.call("fluid.get", domain=DOMAIN),
                         listed_domain(client)):
                assert info["virtual_grains_requested"] == 7
                whitewater = client.call("fluid.get_whitewater", domain=DOMAIN)
                carrier_capacity = info["max_particles"]
                if whitewater["enabled"]:
                    carrier_capacity += whitewater["max_foam"]
                expected_effective = max(
                    1,
                    min(7, 12000000 // max(carrier_capacity, 1)))
                assert info["virtual_grains_effective"] == expected_effective
                assert abs(info["grain_size_variation"] - 0.35) < 1e-6
                assert info["virtual_grain_budget"] == 12000000
                assert info["granular_physical_carriers"] is False
                assert info["primary_visual_children"] == expected_effective
                splat_views = [view for view in info.get("views", [])
                               if view["view"] == "splat"]
                carrier_count = info["particle_count"]
                if info.get("views_measured"):
                    carrier_count = sum(
                        view["particles"] + view["whitewater"]
                        for view in splat_views)
                expected = carrier_count * info["virtual_grains_effective"]
                assert info["virtual_grain_count"] == expected
                assert info["virtual_grain_rt_estimated_bytes"] == expected * 112
            print("[PASS] get/list read-back, stable budget and RT estimate")
        finally:
            client.call("fluid.set_splat_geometry", **restore)
    finally:
        client.close()


if __name__ == "__main__":
    main()
