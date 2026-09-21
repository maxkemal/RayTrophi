"""External production IPC contract test. User launches the application."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import probe_import_export_parity as ipc
from rig_view_contract import check, check_weights, check_binding, check_weight_map, check_multiselection, check_batch_rest, check_mirror, check_pose, check_pose_mirror, check_joint_profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--character")
    parser.add_argument("--mesh", help="Also check read-only canonical flat vertex weights")
    parser.add_argument("--bind-target", help="Explicit mutating bind/undo/redo check on a fresh disposable aligned rig (requires --character)")
    parser.add_argument("--weight-bone", help="Also check selected bone scalar field (requires --character and --mesh)")
    parser.add_argument("--multi-selection", action="store_true", help="Transient multi-selection/pivot check (requires --character)")
    parser.add_argument("--batch-rest", action="store_true", help="Explicit mutating batch rest/undo/redo on disposable owned meshless fixture")
    parser.add_argument("--mirror", action="store_true", help="Explicit mutating mirror test on disposable owned meshless clip-free fixture")
    parser.add_argument("--pose", action="store_true", help="Mutating Pose/keys check on disposable owned bound rig")
    parser.add_argument("--pose-mirror", action="store_true", help="Mutating pose mirror check on disposable owned bound rig with unrestricted anatomy pairs")
    parser.add_argument("--joint-profile", action="store_true", help="Mutating joint rules check on disposable owned bound rig")
    args = parser.parse_args()
    if (args.multi_selection or args.batch_rest or args.mirror or args.pose or args.pose_mirror or args.joint_profile) and not args.character:
        parser.error("Selection/rest checks require --character")
    if args.weight_bone and (not args.character or not args.mesh):
        parser.error("--weight-bone requires --character and --mesh")
    if args.bind_target and not args.character:
        parser.error("--bind-target requires --character for the disposable owned rig")
    pipe = ipc.connect()
    try:
        def invoke(method, params):
            response = ipc.call(pipe, method, params)
            if "error" in response:
                raise RuntimeError(str(response["error"]))
            result = response["result"]
            if isinstance(result, dict) and "__error" in result:
                raise RuntimeError(str(result["__error"]))
            return result
        check(invoke, args.character)
        if args.multi_selection:
            check_multiselection(invoke, args.character)
        if args.batch_rest:
            check_batch_rest(invoke, args.character, lambda: invoke("undo", {}), lambda: invoke("redo", {}))
        if args.mirror:
            check_mirror(invoke, args.character, lambda: invoke("undo", {}), lambda: invoke("redo", {}))
        if args.joint_profile:
            check_joint_profile(invoke, args.character, lambda: invoke("undo", {}), lambda: invoke("redo", {}))
        if args.pose:
            check_pose(invoke, args.character, lambda: invoke("undo", {}), lambda: invoke("redo", {}))
        if args.pose_mirror:
            check_pose_mirror(invoke, args.character, lambda: invoke("undo", {}), lambda: invoke("redo", {}))
        if args.mesh:
            check_weights(invoke, args.mesh)
        if args.weight_bone:
            check_weight_map(invoke, args.mesh, args.character, args.weight_bone)
        if args.bind_target:
            check_binding(invoke, args.character, args.bind_target,
                          lambda: invoke("undo", {}), lambda: invoke("redo", {}))
    finally:
        ipc._kernel32.CloseHandle(pipe)


if __name__ == "__main__":
    main()
