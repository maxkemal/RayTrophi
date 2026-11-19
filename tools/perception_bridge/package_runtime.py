"""Package the currently installed Windows bridge environment without a venv link.

Run using the bridge venv's Python. No pip install or network access is performed.
The destination must not exist; validation precedes publication by the main packager.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig


def package(destination, model):
    destination = Path(destination).resolve()
    model = Path(model).resolve(strict=True)
    source = Path(__file__).resolve().parent
    base = Path(sys.base_prefix).resolve()
    site = Path(sysconfig.get_path("purelib")).resolve()
    if sys.platform != "win32" or sys.version_info[:2] != (3, 10):
        raise RuntimeError("the validated bridge runtime requires Windows Python 3.10")
    for origin in (source, base, site):
        if destination == origin or origin in destination.parents:
            raise ValueError("package destination must be outside the source/runtime trees")
    if destination.exists():
        raise FileExistsError(destination)
    versions = {name: importlib.metadata.version(name)
                for name in ("mediapipe", "opencv-contrib-python", "numpy")}
    # Check native imports before making any output, and again from the copied runtime.
    subprocess.run([sys.executable, "-c", "import mediapipe,cv2,numpy"], check=True)
    runtime = destination / "python"
    runtime.mkdir(parents=True)
    for pattern in ("python.exe", "python3.dll", "python310.dll", "vcruntime*.dll"):
        for file in base.glob(pattern):
            shutil.copy2(file, runtime / file.name)
    if not (runtime / "python.exe").is_file() or not (runtime / "python310.dll").is_file():
        raise RuntimeError("base CPython executable/DLL missing")
    excluded = shutil.ignore_patterns("__pycache__", "*.pyc", "site-packages", "idlelib",
                                     "tkinter", "test", "tests", "ensurepip")
    shutil.copytree(base / "Lib", runtime / "Lib", ignore=excluded)
    shutil.copytree(base / "DLLs", runtime / "DLLs",
                    ignore=shutil.ignore_patterns("*.pdb", "*.lib", "*.exp"))
    shutil.copytree(site, runtime / "site-packages",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "pip", "pip-*",
                                                 "setuptools", "setuptools-*", "*.pth"))
    bridge = destination / "bridge"
    bridge.mkdir()
    for name in ("__init__.py", "bridge.py", "calibration.py", "schema.py", "tracking.py",
                 "transport.py", "video_card.py", "game_bridge.py", "replay_owner_probe.py",
                 "operator_camera.py", "operator_client.py"):
        shutil.copy2(source / name, bridge / name)
    # transport.py uses the established external named-pipe client in scripts/test.
    repo = source.parents[1]
    client = destination / "scripts" / "test"
    client.mkdir(parents=True)
    shutil.copy2(repo / "scripts" / "test" / "rt_ipc.py", client / "rt_ipc.py")
    (runtime / "python310._pth").write_text(
        ".\nLib\nDLLs\nsite-packages\n../bridge\n../scripts/test\n",
        encoding="utf-8")
    models = destination / "models"
    models.mkdir()
    shutil.copy2(model, models / "pose_landmarker.task")
    licences = destination / "licenses"
    licences.mkdir()
    for name in ("LICENSE.txt", "LICENSE"):
        if (base / name).is_file():
            shutil.copy2(base / name, licences / "CPython-LICENSE.txt")
    (licences / "MODEL_SOURCE.txt").write_text(
        "MediaPipe Pose Landmarker task model. Source documentation:\n"
        "https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker\n"
        "Model artifact copied verbatim from the supplied local model.\n"
        "Python distributions retain their bundled licenses and .dist-info notices.\n",
        encoding="utf-8")
    subprocess.run([str(runtime / "python.exe"), "-I", "-c",
                    "import cv2,mediapipe,numpy,bridge,game_bridge,transport,operator_camera,operator_client; "
                    "print('portable bridge imports PASS')"], cwd=destination, check=True)
    manifest = {"python": sys.version.split()[0], "packages": versions,
                "image_storage": "memory_only", "model_sha256": hashlib.sha256(
                    model.read_bytes()).hexdigest()}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.output, args.model)))
