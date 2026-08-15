"""Stamp e-on/Bentley licensing metadata onto vegetation asset descriptors.

The plants under assets/vegetation come from PlantCatalog via PlantFactory and
carry a permanent no-sale restriction (EULA section 2.2). AssetRegistry reads
`license` and `source` from each `*.asset.json` and writes them back unchanged,
so stamping them once is durable.

Scope is deliberately limited to assets/vegetation. Volume/VDB assets have a
different, still-unrecorded provenance and must not be labeled with these terms.

Usage:
    python scripts/stamp_vegetation_licenses.py <assets_dir>          # dry run
    python scripts/stamp_vegetation_licenses.py <assets_dir> --write

See assets/THIRD_PARTY_ASSETS.md for the governing clause.
"""

import argparse
import json
import pathlib
import sys

LICENSE = "e-on/Bentley EULA 2.2 - PlantCatalog derivative; sharing permitted, SALE FORBIDDEN"
SOURCE = "PlantCatalog (via PlantFactory export)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets_dir", help="path to an assets/ directory")
    parser.add_argument("--write", action="store_true",
                        help="apply changes (default is a dry run)")
    args = parser.parse_args()

    vegetation = pathlib.Path(args.assets_dir) / "vegetation"
    if not vegetation.is_dir():
        print("no vegetation directory under: %s" % args.assets_dir, file=sys.stderr)
        return 1

    descriptors = sorted(vegetation.rglob("*.asset.json"))
    if not descriptors:
        print("no *.asset.json found under %s" % vegetation, file=sys.stderr)
        return 1

    changed = 0
    for path in descriptors:
        try:
            with path.open(encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError) as error:
            print("SKIP %s (%s)" % (path, error), file=sys.stderr)
            continue

        before = (data.get("license"), data.get("source"))
        after = (LICENSE, SOURCE)
        if before == after:
            continue

        rel = path.relative_to(vegetation)
        print("%s\n    license: %s -> %s\n    source:  %s -> %s"
              % (rel, before[0], after[0], before[1], after[1]))
        changed += 1

        if args.write:
            data["license"] = LICENSE
            data["source"] = SOURCE
            with path.open("w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, sort_keys=True, ensure_ascii=False)
                handle.write("\n")

    total = len(descriptors)
    if not args.write:
        print("\ndry run: %d/%d descriptors would change (pass --write to apply)"
              % (changed, total))
    else:
        print("\nstamped %d/%d descriptors" % (changed, total))

    # Models without a descriptor are invisible to this stamping pass and will
    # keep reporting license "unknown" from AssetRegistry's auto-discovery.
    models = {p.with_suffix("").name for p in vegetation.rglob("*.glb")}
    described = {p.name[:-len(".asset.json")] for p in descriptors}
    orphans = sorted(models - described)
    if orphans:
        print("\nWARNING: %d model(s) have no .asset.json and stay unlabeled:"
              % len(orphans))
        for name in orphans:
            print("    %s" % name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
