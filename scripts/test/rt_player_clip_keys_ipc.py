"""Manual keyboard acceptance bridge, not a native Player key binding.

Space toggles two prepared clips only while this IPC server's Player window is
foreground. F8 ends the bridge. No keyboard hook, text capture, scene save, app
launch or C++ build. Leaves the selected clip playing when the bridge finishes.
"""
import argparse
import ctypes
from ctypes import wintypes
import json
import math
from pathlib import Path
import time

from rt_ipc import PLAYER_PIPE_NAME, RtIpc
from rt_player_clip_transitions_ipc import wait_clip


class PlayerKeys:
    def __init__(self, client):
        self.user = ctypes.WinDLL("user32", use_last_error=True)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetNamedPipeServerProcessId.argtypes = [wintypes.HANDLE,
                                                       ctypes.POINTER(wintypes.ULONG)]
        kernel.GetNamedPipeServerProcessId.restype = wintypes.BOOL
        pid = wintypes.ULONG()
        if not kernel.GetNamedPipeServerProcessId(client.handle, ctypes.byref(pid)):
            raise ctypes.WinError(ctypes.get_last_error())
        self.pid = pid.value
        self.user.GetForegroundWindow.restype = wintypes.HWND
        self.user.GetWindowThreadProcessId.argtypes = [wintypes.HWND,
                                                       ctypes.POINTER(wintypes.DWORD)]
        self.user.GetAsyncKeyState.argtypes = [ctypes.c_int]
        self.user.GetAsyncKeyState.restype = ctypes.c_short
        self.focused = False
        self.down = {0x20: False, 0x77: False}

    def poll(self):
        pid = wintypes.DWORD()
        window = self.user.GetForegroundWindow()
        self.user.GetWindowThreadProcessId(window, ctypes.byref(pid))
        if pid.value != self.pid:
            self.focused = False
            return set()
        pressed = set()
        for key in self.down:
            down = bool(self.user.GetAsyncKeyState(key) & 0x8000)
            if self.focused and down and not self.down[key]:
                pressed.add(key)
            self.down[key] = down
        self.focused = True
        return pressed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--character", default="1")
    parser.add_argument("--duration", type=float, default=300)
    parser.add_argument("--blend", type=float, default=0.65)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert math.isfinite(args.duration) and 1 <= args.duration <= 3600
    assert math.isfinite(args.blend) and 0 <= args.blend <= 5
    report = dict(status="STARTING", mode="external_keyboard_ipc_test",
                  character=args.character, events=[])

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    client = RtIpc(PLAYER_PIPE_NAME)
    try:
        runtime = client.call("agent.get_state_summary")["runtime"]
        assert runtime["scene_ready"], "load the prepared two-clip project first"
        clips = client.call("anim.clips", character=args.character)
        assert len(clips) == 2, "test expects exactly two clips"
        names = [clip["name"] for clip in clips]
        keys = PlayerKeys(client)
        client.call("anim.play", character=args.character, clip=names[0], blend=0)
        client.call("anim.set_speed", character=args.character, speed=1)
        wait_clip(client, args.character, names[0])
        current = 0
        report.update(status="LISTENING", clips=names, current_clip=names[0],
                      blend_seconds=args.blend,
                      crossfade_tested=args.blend > 0)
        save()
        print("READY: Player foreground + SPACE = toggle clips; F8 = end test.", flush=True)
        print(f"A={names[0]}  B={names[1]}  blend={args.blend}s", flush=True)
        started = time.monotonic()
        while time.monotonic() - started < args.duration:
            pressed = keys.poll()
            if 0x77 in pressed:
                break
            if 0x20 in pressed:
                target = 1 - current
                event = dict(from_clip=names[current], to_clip=names[target],
                             elapsed_s=time.monotonic() - started)
                report["events"].append(event)
                client.call("anim.play", character=args.character,
                            clip=names[target], blend=args.blend)
                status, blended = wait_clip(client, args.character, names[target],
                                            timeout=max(6, args.blend + 2))
                if args.blend > 0:
                    assert blended, "crossfade was not observed"
                assert status["playing"] and not status["paused"], status
                runtime = client.call("agent.get_state_summary")["runtime"]
                assert not runtime["device_lost"] and not runtime["error"]
                assert not runtime["simulation"]["error"]
                current = target
                event.update(status="PASS", blending_seen=blended)
                report["current_clip"] = names[current]
                save()
                print(f"PASS: {event['from_clip']} -> {names[current]}", flush=True)
            time.sleep(0.015)
        report["status"] = "PASS" if report["events"] else "NOT_RUN"
    except Exception as error:
        report.update(status="FAIL", error=str(error))
        raise
    finally:
        client.close()
        save()
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
