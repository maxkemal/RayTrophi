"""External Player two-clip acceptance. Load a disposable prepared project first.

Uses shared animation APIs and real loop-driven joint snapshots. Never launches
or builds the app, saves the scene, or changes frame capture. Reopens the same
project at the end to restore its authored playback (resets runtime simulation).
Keep the Player foreground while running; use an outer process timeout.
"""
import argparse
import json
import math
import time
from pathlib import Path

from rt_ipc import PLAYER_PIPE_NAME, RtIpc, RtIpcError


def joint_distance(first, last):
    assert first and first.keys() == last.keys(), "skeleton membership changed"
    return max(math.dist(first[name], last[name]) for name in first)


def bones(client, character):
    rows = client.call("rig.list_bones", character=character)
    return {row["name"]: row["world_position"] for row in rows}


def wait_clip(client, character, clip, timeout=6):
    deadline = time.monotonic() + timeout
    blending_seen = False
    while time.monotonic() < deadline:
        status = client.call("anim.status", character=character)
        blending_seen |= status["blending"]
        if status["clip"] == clip and not status["blending"]:
            return status, blending_seen
        time.sleep(0.04)
    raise AssertionError(f"clip did not settle: {status}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--character", default="1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    client = RtIpc(PLAYER_PIPE_NAME)
    report = dict(status="FAIL", character=args.character, transitions=[])
    project = None
    changed = False
    try:
        first = client.call("agent.get_state_summary")["runtime"]
        assert first["scene_ready"] and first["project_path"], "load prepared project"
        project = first["project_path"]
        clips = client.call("anim.clips", character=args.character)
        assert len(clips) >= 2, "prepare a project with at least two bound clips"
        a, b = (clip["name"] for clip in clips[:2])
        report["clips"] = clips
        before = client.call("anim.character", character=args.character)
        for params in [dict(clip="__missing_clip__"), dict(clip=a, layer=4),
                       dict(clip=a, blend=-1)]:
            try:
                client.call("anim.play", character=args.character, **params)
            except RtIpcError:
                pass
            else:
                raise AssertionError(f"invalid play accepted: {params}")
        assert client.call("anim.character", character=args.character) == before
        changed = True
        client.call("anim.play", character=args.character, clip=a, blend=0)
        client.call("anim.set_speed", character=args.character, speed=1)
        wait_clip(client, args.character, a)
        for clip in (b, a):
            client.call("anim.play", character=args.character, clip=clip, blend=0.8)
            _, blended = wait_clip(client, args.character, clip)
            assert blended, "controller never reported an active crossfade"
            assert not client.call("anim.character", character=args.character)["uses_graph"]
            snapshots = [bones(client, args.character)]
            for _ in range(12):
                time.sleep(0.07)
                snapshots.append(bones(client, args.character))
            excursion = max(joint_distance(snapshots[0], s) for s in snapshots[1:])
            assert excursion > 0.001, f"no visible joint progression for {clip}"
            report["transitions"].append(dict(clip=clip, excursion_m=excursion,
                                              blending_seen=blended))
        client.call("anim.set_paused", character=args.character, paused=True)
        time.sleep(0.15)
        held = bones(client, args.character)
        held_status = client.call("anim.status", character=args.character)
        time.sleep(0.25)
        assert joint_distance(held, bones(client, args.character)) < 1e-6
        assert client.call("anim.status", character=args.character)["time"] == held_status["time"]
        client.call("anim.set_paused", character=args.character, paused=False)
        time.sleep(0.15)
        assert joint_distance(held, bones(client, args.character)) > 0.001
        last = client.call("agent.get_state_summary")["runtime"]
        assert not last["device_lost"] and not last["error"]
        assert not last["simulation"]["error"]
        assert last["animation_updates"] > first["animation_updates"]
        assert last["animation_skin_sync_calls"] > first["animation_skin_sync_calls"]
        client.call("anim.stop", character=args.character, blend_out=0)
        assert not client.call("anim.status", character=args.character)["playing"]
        report.update(status="PASS", runtime_before=first, runtime_after=last)
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        try:
            if changed and project:
                client.call("project.open", path=project)
                report["authored_project_reopened"] = True
        except Exception as error:
            report.update(status="FAIL", restore_error=str(error))
            raise
        finally:
            client.close()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
