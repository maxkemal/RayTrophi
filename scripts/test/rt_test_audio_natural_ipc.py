"""Post-build V2 contracts; --exercise requires an empty disposable audio scene.

No builds, launch, solver edits, telemetry enabling or playback. Exercise creates
temporary sources and dry bakes, removes them, and restores transient workspace.
"""
import argparse
import math
from pathlib import Path
import tempfile
import time

from rt_ipc import RtIpc, RtIpcError

MODELS = ("liquid_flow", "gas_flow", "friction", "rolling", "impact",
          "contact_activity", "deformation", "fracture_activity")


def reject(client, method, **params):
    try:
        client.call(method, **params)
    except RtIpcError:
        return
    raise AssertionError(f"Invalid request accepted: {method} {params}")


def bake(client, source, path):
    client.call("audio.physical_bake_start", id=source, path=str(path))
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status = client.call("audio.physical_bake_status")
        assert status["state"] not in ("failed", "cancelled", "stale"), status
        if status["state"] == "completed":
            return client.call("audio.physical_bake_apply")
        time.sleep(0.05)
    raise AssertionError("Bake did not finish")


def exercise(client):
    initial = client.call("audio.list")
    assert not any(initial[key] for key in ("emitters", "assets", "clips", "tracks")), (
        "Use an empty disposable audio scene; authoring creates scene history")
    with tempfile.TemporaryDirectory(prefix="rt_audio_v2_") as folder:
        for model in MODELS:
            source = None
            assets = []
            try:
                source = client.call("audio.create", name=f"V2 {model}", gain=0.15)["id"]
                configured = client.call("audio.physical_configure", id=source,
                                         generator={"version": 2, "character_revision": 2, "model": model})
                assert configured["id"] == source and math.isclose(configured["gain"], 0.15, rel_tol=1e-6)
                assert configured["asset_id"] == "" and configured["loop"]
                assert configured["generator"]["version"] == 2
                assert configured["generator"]["character_revision"] == 2
                assert all(configured["generator"][key] == 0.5
                           for key in ("texture", "variation", "body"))
                stable = client.call("audio.get", id=source)
                invalid = [{"version": 99, "model": model},
                           {"version": 1, "model": model, "texture": 0.5},
                           {"version": 2, "model": model, "unknown": 1}]
                for key in ("texture", "variation", "body"):
                    invalid.extend({"version": 2, "model": model, key: value}
                                   for value in (-0.1, 1.1, True, None, "0.5"))
                invalid.extend({"version": 2, "character_revision": value, "model": model}
                               for value in (0, 3, None, True, "2"))
                invalid.append({"version": 1, "model": model, "character_revision": 2})
                invalid.append({"version": 2, "model": model, "surface": "organic"})
                if model == "friction":
                    invalid.extend({"version": 2, "character_revision": 2, "model": model,
                                    "surface": value} for value in (None, True, "invalid", 1))
                for value in (None, 1, "yes"):
                    reject(client, "audio.physical_configure", id=source,
                           generator=stable["generator"], clear_history=value)
                    assert client.call("audio.get", id=source) == stable
                for recipe in invalid:
                    reject(client, "audio.physical_configure", id=source, generator=recipe)
                    assert client.call("audio.get", id=source) == stable
                history = {"version": 1, "duration_seconds": 0.25}
                if model == "impact":
                    history["events"] = [{"frame": 0, "sequence": 1, "energy_j": 0.1}]
                else:
                    history["controls"] = [{"frame": 0, "activity": 0.8},
                                           {"frame": 9000, "activity": 0}]
                client.call("audio.update", id=source, generator_history=history)
                hashes = []
                for index in range(2):
                    result = bake(client, source, Path(folder) / f"{model}_{index}.wav")
                    assets.append(result["asset"]["id"])
                    assert result["bake"]["frames"] == 12000
                    assert math.isfinite(result["bake"]["peak"]) and result["bake"]["peak"] > 0
                    assert not result["bake"]["acoustic_processing"]
                    hashes.append(result["bake"]["pcm_fnv1a64"])
                assert hashes[0] == hashes[1], model
                def revision_hash(recipe, suffix):
                    client.call("audio.physical_configure", id=source, generator=recipe)
                    result = bake(client, source, Path(folder) / f"{model}_{suffix}.wav")
                    assets.append(result["asset"]["id"])
                    return result["bake"]["pcm_fnv1a64"]

                original = dict(configured["generator"], character_revision=1)
                original.pop("surface", None)
                original_hash = revision_hash(original, "original_character")
                if model in ("liquid_flow", "gas_flow", "friction", "impact", "fracture_activity"):
                    assert original_hash != hashes[0], model
                else:
                    assert original_hash == hashes[0], model
                if model == "impact":
                    for material in ("wood", "stone", "metal", "glass"):
                        left = revision_hash(dict(original, resonator=material), material + "_r1")
                        right = revision_hash(dict(original, resonator=material, character_revision=2),
                                              material + "_r2")
                        assert (left == right) == (material in ("metal", "glass")), material
                if model == "friction":
                    surfaces = {revision_hash(dict(configured["generator"], surface=surface), surface)
                                for surface in ("organic", "mineral", "metal", "glass")}
                    assert len(surfaces) == 4
                client.call("audio.physical_configure", id=source, generator=configured["generator"])
                changed = dict(configured["generator"], body=0.95, texture=0.05)
                client.call("audio.physical_configure", id=source, generator=changed)
                result = bake(client, source, Path(folder) / f"{model}_changed.wav")
                assets.append(result["asset"]["id"])
                assert result["bake"]["pcm_fnv1a64"] != hashes[0], model
                legacy = client.call("audio.physical_configure", id=source,
                                     generator={"version": 1, "model": model})
                assert legacy["generator"]["version"] == 1
                assert "texture" not in legacy["generator"]
                result = bake(client, source, Path(folder) / f"{model}_legacy.wav")
                assets.append(result["asset"]["id"])
                assert result["bake"]["pcm_fnv1a64"] != hashes[0], model
                if model == "impact":
                    before = client.call("audio.get", id=source)
                    reject(client, "audio.physical_configure", id=source,
                           generator={"version": 2, "model": "liquid_flow"})
                    assert client.call("audio.get", id=source) == before
                    cleared = client.call("audio.physical_configure", id=source,
                                          generator={"version": 2, "model": "liquid_flow"},
                                          clear_history=True)
                    assert cleared["generator_history"] is None
                print(f"PASS {model}: version, atomic errors, deterministic bake, controls, legacy")
            finally:
                client.call("audio.physical_bake_cancel")
                if source is not None:
                    client.call("audio.remove", id=source)
                for asset in assets:
                    client.call("audio.asset_remove", id=asset)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exercise", action="store_true")
    args = parser.parse_args()
    client = RtIpc()
    workspace = client.call("audio.workspace_get")
    try:
        catalogue = client.call("audio.physical_catalogue")
        assert catalogue["versions"] == [1, 2] and catalogue["legacy_preserved"]
        assert catalogue["character_revisions"] == [1, 2]
        assert catalogue["default_character_revision"] == 1
        assert catalogue["recommended_character_revision"] == 2
        assert catalogue["friction_surfaces"] == ["organic", "mineral", "metal", "glass"]
        assert catalogue["quality_scope"] == "artistic_activity_sonification"
        assert not catalogue["measured_discrete_events"]
        assert tuple(entry["model"] for entry in catalogue["models"]) == MODELS
        client.call("audio.workspace_update", designer_open=False)
        for scope in ("design", "scene"):
            assert client.call("audio.workspace_update", history_scope=scope)["history_scope"] == scope
        stable = client.call("audio.workspace_get")
        for scope in (None, True, 1, "invalid"):
            reject(client, "audio.workspace_update", history_scope=scope, mixer_bus="Music")
            assert client.call("audio.workspace_get") == stable
        reject(client, "audio.physical_catalogue", unknown=True)
        print("PASS catalogue and focus-scope contracts")
        if args.exercise:
            exercise(client)
    finally:
        client.call("audio.workspace_update", **workspace)
        client.close()


if __name__ == "__main__":
    main()
