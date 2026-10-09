"""One external-process command for the implemented Matter acceptance gates.

Run after the final user build, in an empty paused scene, with no other IPC test
running. Extended H1 fixtures remain disabled for inspection, as in that suite.
No build or app launch. Regression PASS does not close remaining porous,
angular, thermal or production-scale acceptance gates.
"""

import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

from rt_ipc import RtIpc


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "scripts" / "test"


def require_empty_paused_scene(empty_required=True):
    client = RtIpc()
    try:
        assert not client.call("sim.control_state")["playing"], "Pause the timeline"
        if empty_required:
            assert not client.call("fluid.list_domains")["domains"], "Use an empty scene"
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extended", action="store_true",
                        help="Include dry DEM static, timestep, repose and fluid regressions")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or ROOT / "docs" / "dev" / f"matter_acceptance_{stamp}.json"
    jobs = [
        ("substance", "rt_test_substance_profiles_ipc.py", []),
        ("domain_material", "rt_test_domain_substance_ipc.py", []),
        ("sparse_pressure", "rt_test_sparse_pressure_ipc.py", []),
        ("sparse_viscosity", "rt_test_sparse_pressure_ipc.py", ["--viscosity"]),
        ("sparse_mac_transfer", "rt_test_sparse_pressure_ipc.py", ["--transfer"]),
        ("sparse_mac_transfer_viscosity", "rt_test_sparse_pressure_ipc.py",
         ["--transfer", "--viscosity"]),
        ("shared_clock_and_dynamic_support", "rt_test_matter_transport_ipc.py",
         ["--kernel-timings"]),
    ]
    if args.extended:
        for name, flag in [
            ("readiness", "--readiness-only"),
            ("history", "--history-only"),
            ("static_contact", "--static-only"),
            ("dt_convergence", "--convergence-only"),
            ("repose", "--repose-only"),
            ("liquid_coexistence", "--coexist-only"),
            ("porous_two_owner_regression", "--porous-only"),
        ]:
            jobs.append((name, "rt_h1_grain_runtime_ipc.py", [flag]))
    report = {
        "started_utc": stamp,
        "status": "running",
        "all_unified_matter_gates_closed": False,
        "remaining_gates": [
            "three_owner_porous_projection_and_pressure_reaction",
            "mpm_grain_deformation_and_angular_convergence",
            "cinematic_scale_native_gpu_cost_and_boundary_work",
            "complete_sparse_mac_gas_storage_and_projection",
            "thermal_energy_and_gas_elastic_acceptance",
        ],
        "tests": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        require_empty_paused_scene()
        for name, script, flags in jobs:
            require_empty_paused_scene(script != "rt_h1_grain_runtime_ipc.py")
            print(f"RUN {name}", flush=True)
            started = time.perf_counter()
            result = subprocess.run([sys.executable, "-u", str(TESTS / script), *flags],
                                    cwd=ROOT, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace")
            measurements = []
            for line in result.stdout.splitlines():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    measurements.append(value)
            record = {
                "name": name,
                "returncode": result.returncode,
                "elapsed_wall_seconds": time.perf_counter() - started,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "measurements": measurements,
            }
            if script == "rt_h1_grain_runtime_ipc.py":
                source = ROOT / "docs" / "dev" / "matter_h1_grain_runtime_live.json"
                if source.exists():
                    snapshot = output.with_name(output.stem + "_" + name + ".json")
                    snapshot.write_bytes(source.read_bytes())
                    record["detail_report"] = str(snapshot)
            report["tests"].append(record)
            save()
            print(result.stdout, end="", flush=True)
            if result.returncode:
                print(result.stderr, end="", file=sys.stderr, flush=True)
                report["status"] = "failed"
                return 1
            require_empty_paused_scene(script != "rt_h1_grain_runtime_ipc.py")
        report["status"] = "implemented_gates_passed"
        print(f"PASS implemented gates; full physics/cinematic gates remain open. Report: {output}")
        return 0
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        return 130
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    raise SystemExit(main())
