"""Read-only post-build gas-source publication probe. Run externally with Studio open."""
import argparse
import json
from rt_ipc import RtIpc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--expect-source", action="append", default=[])
    args = parser.parse_args()
    client = RtIpc()
    try:
        inspection = client.call("fluid.inspect_fog_fields", domain=args.domain)
        fields = inspection["fields"]
        gas = [field for field in fields if field["kind"] in ("gas_source", "gas_aggregate")]
        assert gas, "No gas fields: initialize/resimulate the Matter domain first"
        assert inspection["gas_max_abs_density_error"] <= 1e-4, "Gas density partition mismatch"
        targets = {row["target"] for row in client.call("volume.list_materials")["targets"]}
        ids = set()
        for field in gas:
            assert field["material_target"] in targets, field
            material = client.call("volume.get_material", target=field["material_target"])
            assert isinstance(material["visible"], bool), material
            if field["active_cells"] > 0:
                assert field["published"], field
                assert field["vdb_id"] >= 0 and field["vdb_id"] not in ids, field
                ids.add(field["vdb_id"])
        sources = client.call("flow_source.list")
        for name in args.expect_source:
            matches = [source for source in sources if source["name"] == name
                       and source["domain"] == args.domain and source["phase"] == "gas"]
            assert len(matches) == 1, f"Missing or ambiguous gas source: {name}"
            uid = matches[0]["producer_uid"]
            assert uid > 0
            matches = [field for field in gas if field["kind"] == "gas_source"
                       and field["producer_uid"] == uid]
            assert len(matches) == 1, f"No source field for {name}: resimulate after emission"
        print(json.dumps(inspection, ensure_ascii=False, indent=2))
        print("PASS: source identities, density partition, publication and material readback")
    finally:
        client.close()


if __name__ == "__main__":
    main()
