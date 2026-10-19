"""Post-build substance catalogue/inheritance acceptance; run externally with Studio open.

Does not launch Studio, save the scene, or change existing profiles/assignments.
Creates and removes two uniquely named project profiles.
"""
import uuid

from rt_ipc import RtIpc, RtIpcError


FOUNDATIONS = {
    "Generic metal": ("Steel", "metals"),
    "Generic mineral": ("Stone", "minerals"),
    "Generic polymer": ("Plastic (PE)", "polymers"),
    "Generic fibrous solid": ("Wood (Oak)", "fibrous"),
    "Generic liquid": ("Water", "liquids"),
    "Generic mineral grains": ("Sand", "minerals"),
}


def main():
    client = RtIpc()
    suffix = uuid.uuid4().hex[:12]
    parent, child = f"Catalogue_parent_{suffix}", f"Catalogue_child_{suffix}"
    created = []
    try:
        listing = client.call("substance.list")["substances"]
        before_names = [row["name"] for row in listing]
        rows = {row["name"]: row for row in listing}
        assert len(rows) == len(listing), "duplicate names"
        foundations = {row["name"] for row in listing
                       if row["catalogue"]["tier"] == "foundation"}
        assert foundations == set(FOUNDATIONS), foundations
        for name, (representative, family) in FOUNDATIONS.items():
            value = client.call("substance.get", name=name)
            example = client.call("substance.get", name=representative)
            assert value["fields"] == example["fields"], name
            assert value["catalogue"] == rows[name]["catalogue"], name
            assert value["catalogue"]["family"] == family
            assert value["catalogue"]["representative"] == representative
            assert value["edit_scope"] == "read_only_builtin"
            assert rows[representative]["catalogue"]["tier"] == "example"
        assert rows["Flesh"]["catalogue"]["tier"] == "test"
        client.call("substance.derive", name=parent, based_on="Generic metal")
        created.append(parent)
        client.call("substance.derive", name=child, based_on=parent)
        created.append(child)
        derived = client.call("substance.get", name=child)
        assert derived["catalogue"]["tier"] == "project"
        assert derived["catalogue"]["family"] == "metals"
        assert derived["catalogue"]["root_profile"] == "Generic metal"
        assert derived["edit_scope"] == "shared_project_profile"
        affected = client.call("substance.get", name=parent)["affected_references"]
        assert f"derived profile '{child}'" in affected
        density = derived["fields"]["density"] * 1.01
        client.call("substance.set", name=parent, fields={"density": density})
        assert client.call("substance.get", name=child)["fields"]["density"] == density
        stable = client.call("substance.get", name=parent)
        for invalid in ({"family": "metals"}, {"density": -1}):
            try:
                client.call("substance.set", name=parent, fields=invalid)
            except RtIpcError:
                pass
            else:
                raise AssertionError(f"invalid patch accepted: {invalid}")
            assert client.call("substance.get", name=parent) == stable
        client.call("substance.set", name=parent, fields={"density": None})
        assert client.call("substance.get", name=child)["fields"] == derived["fields"]
        for name in reversed(created):
            client.call("substance.remove", name=name)
        created.clear()
        assert [row["name"] for row in client.call("substance.list")["substances"]] == before_names
        print("PASS: catalogue, representative fields, inherited family, shared scope, atomic errors")
    finally:
        for name in reversed(created):
            client.call("substance.remove", name=name)
        client.close()


if __name__ == "__main__":
    main()
