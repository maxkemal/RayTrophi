"""Post-build disposable-scene test. External IPC only; does not build or launch Studio."""
import math
import uuid
from rt_ipc import RtIpc, RtIpcError


def main():
    client = RtIpc()
    prefix = "Placement_" + uuid.uuid4().hex[:8]
    domains = []
    source = None

    def create(suffix, **params):
        result = client.call("fluid.create_placed_domain", name=prefix + suffix, **params)
        domains.append(result["domain"])
        return result

    def close_vec(actual, expected):
        assert all(math.isclose(a, b, abs_tol=1e-5) for a, b in zip(actual, expected)), (actual, expected)

    try:
        client.call("timeline.pause")
        source = client.call("scene.add_primitive", type="cube", name=prefix + "Mesh", size=2)
        client.call("scene.set_transform", name=source, translation=[7, 3, -4],
                    rotation=[13, 27, 41], scale=[1, 2, .5])
        info = client.call("scene.object_info", name=source)
        assert info["meshes"] > 0 and info["has_bounds"], info
        client.call("select.object", name=source)
        selected = create("SelectedDefault")
        client.call("select.clear")
        unselected = create("UnselectedDefault")
        for row in (selected, unselected):
            assert row["placement"] == "ground_default"
            close_vec(row["bounds_min"], [-2.5, 0, -2.5])
            close_vec(row["bounds_max"], [2.5, 5, 2.5])
        fitted = create("Fitted", source_object=source)
        close_vec(fitted["bounds_min"], info["world_min"])
        close_vec(fitted["bounds_max"], info["world_max"])
        client.call("scene.set_transform", name=source, translation=[-8, 4, 6])
        moved = client.call("scene.object_info", name=source)
        fitted_moved = create("Moved", source_object=source)
        close_vec(fitted_moved["bounds_min"], moved["world_min"])
        close_vec(fitted_moved["bounds_max"], moved["world_max"])
        before = client.call("fluid.list_domains")
        for params in ({"name": prefix + "Missing", "source_object": prefix + "Absent"},
                       {"name": selected["domain"]}):
            try:
                client.call("fluid.create_placed_domain", **params)
            except RtIpcError:
                pass
            else:
                raise AssertionError("accepted invalid placement: " + repr(params))
        after = client.call("fluid.list_domains")
        # Telemetry may change between calls; domain identity must not.
        def names(rows):
            if isinstance(rows, dict):
                rows = rows.get("domains", rows)
            return sorted(row["name"] for row in rows)
        assert names(before) == names(after)
        print("PASS selection-independent ground defaults, transformed flat mesh fitting, "
              "fresh transforms and transactional rejection")
    finally:
        try:
            for domain in reversed(domains):
                client.call("fluid.remove_domain", domain=domain)
            if source:
                client.call("scene.delete", name=source)
        finally:
            client.close()


if __name__ == "__main__":
    main()
