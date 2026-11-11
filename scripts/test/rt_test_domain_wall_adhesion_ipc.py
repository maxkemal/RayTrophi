"""Run externally AFTER user C++ + shaders build; requires a paused empty scene.

python scripts/test/rt_test_domain_wall_adhesion_ipc.py [--backend vulkan|cpu]
Creates/removes a domain; checks settings and a free-vs-glued ceiling slab.
"""
import argparse
from rt_ipc import RtIpc, RtIpcError

DOMAIN = "T_DomainWallAdhesion"


def main(backend):
    call = RtIpc().call
    assert not call("sim.control_state")["playing"], "Pause first"
    assert not call("fluid.list_domains")["domains"], "Use an empty scene"
    made = False
    try:
        call("fluid.create_domain", name=DOMAIN, type="fluid",
             domain_min=[0, 0, 0], domain_max=[.2, .2, .2], voxel_size=.01)
        made = True
        call("fluid.set_param", domain=DOMAIN, enabled=True, visible=False,
             backend=backend, boundary="closed", default_substance="Water",
             thermal_liquid_enabled=False)
        call("fluid.settle", domain=DOMAIN, on_seed=False, request=False)
        assert call("fluid.get_wall_contact", domain=DOMAIN)["adhesion_pa"] == 0
        expected = call("fluid.set_wall_contact", domain=DOMAIN, settings={
            "friction": .3, "restitution": 0, "adhesion_pa": 1234})
        for bad in (-1, 1e10, True, None, "sticky"):
            try:
                call("fluid.set_wall_contact", domain=DOMAIN,
                     settings={"friction": .8, "adhesion_pa": bad})
            except RtIpcError:
                pass
            else:
                raise AssertionError("invalid adhesion accepted: " + repr(bad))
            assert call("fluid.get_wall_contact", domain=DOMAIN) == expected
        call("fluid.set_param", domain=DOMAIN, boundary="open")
        inactive = call("fluid.get_wall_contact", domain=DOMAIN)
        assert not inactive["active"] and inactive["adhesion_pa"] == 1234
        call("fluid.set_param", domain=DOMAIN, boundary="closed")

        def arm(adhesion):
            call("fluid.set_wall_contact", domain=DOMAIN, settings={
                "friction": 0, "restitution": 0, "adhesion_pa": adhesion})
            call("timeline.set_frame", frame=0)
            call("fluid.seed", domain=DOMAIN, seed_min=[.05, .18, .05],
                 seed_max=[.15, .195, .15], particles_per_cell=8, replace=True)
            grid = call("fluid.sample_solid_phi", domain=DOMAIN, points=[[.1, .19, .1]])
            start = call("fluid.state_digest", domain=DOMAIN)
            assert start["particles"] > 0
            for _ in range(10):
                call("fluid.step", dt=.01)
            end = call("fluid.state_digest", domain=DOMAIN)
            assert end["particles"] == start["particles"]
            return start["bounds_max"][1] - end["bounds_max"][1], grid["voxel_m"]

        free_drop, h = arm(0)
        glued_drop, _ = arm(1e6)
        print(backend, "ceiling drop (free, glued, voxel):", free_drop, glued_drop, h,
              flush=True)
        assert free_drop > h, "free slab did not separate from ceiling"
        assert glued_drop < .5 * h, "adhesive ceiling failed to retain upper slab"
        assert free_drop - glued_drop > h, "no measurable adhesion effect"
        print("PASS adhesion settings, atomic validation, open boundary, ceiling A/B")
    finally:
        if made:
            call("fluid.remove_domain", domain=DOMAIN)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("vulkan", "cpu"), default="vulkan")
    main(parser.parse_args().backend)
