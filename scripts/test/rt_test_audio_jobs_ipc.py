"""Post-build background Designer acceptance; run externally with Studio open.

No files or scene assets are created, and nothing is played. Reuses the previous
Designer recipe when present; an initially empty Designer retains the test recipe.
Do not edit the Designer during this ownership/PCM determinism probe.
"""
import time
from rt_ipc import RtIpc


def main():
    client = RtIpc()
    previous = client.call("audio.design_status")
    scene = client.call("audio.list")
    workspace = client.call("audio.workspace_get")
    recipe = {"mode": "synth", "preset": "tone", "frequency_hz": 440,
              "duration_seconds": 0.2, "gain": 0.1, "seed": 17}

    def wait(identity):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            status = client.call("audio.design_job_status")
            assert status["id"] == identity, status
            if status["state"] == "completed":
                return
            assert status["state"] in ("queued", "running"), status
            time.sleep(0.02)
        raise AssertionError("Background job timed out")

    def start(**params):
        return client.call("audio.design_job_start", **params)["id"]

    try:
        client.call("audio.workspace_update", designer_open=False)
        client.call("audio.design_render", recipe=recipe)
        reference = client.call("audio.design_analyze", fft_size=512,
                                points=16, time_bins=4)
        revision = client.call("audio.design_status", details=False)["revision"]
        first = start(kind="render", recipe={**recipe, "frequency_hz": 880})
        last = start(kind="render", recipe=recipe)
        assert last > first
        wait(last)
        assert client.call("audio.design_status", details=False)["revision"] == revision
        result = client.call("audio.design_job_apply")
        assert result["id"] == last and result["state"] == "applied", result
        assert result["design"]["recipe"]["frequency_hz"] == 440
        assert client.call("audio.design_analyze", fft_size=512,
                           points=16, time_bins=4)["pcm_hash"] == reference["pcm_hash"]

        identity = start(kind="analysis", options={"fft_size": 512,
                         "points": 16, "time_bins": 4})
        wait(identity)
        analysis = client.call("audio.design_job_apply")["analysis"]
        for field in ("pcm_hash", "spectrum", "spectrogram", "channel_stats"):
            assert analysis[field] == reference[field], field

        start(kind="render", recipe={**recipe, "duration_seconds": 10})
        client.call("audio.design_job_cancel")
        assert client.call("audio.design_job_apply")["state"] == "cancelled"
        assert client.call("audio.design_status", details=False)["revision"] == revision + 1

        before = client.call("audio.design_job_status")
        for params in ({"kind": "unknown"},
                       {"kind": "render", "recipe": {"unknown": 1}},
                       {"kind": "analysis", "options": {"fft_size": 513}},
                       {"kind": "analysis", "options": {"unknown": 1}},
                       {"kind": "render", "recipe": recipe, "options": {}}):
            accepted, error = client.try_call("audio.design_job_start", **params)
            assert not accepted, (params, error)
            assert client.call("audio.design_job_status") == before
        assert client.call("audio.list") == scene
        print("PASS: latest job ownership, explicit apply, PCM/analysis parity, "
              "cancellation, atomic validation and scene isolation")
    finally:
        client.call("audio.design_job_cancel")
        if previous.get("ready"):
            client.call("audio.design_render", recipe=previous["recipe"])
        client.call("audio.workspace_update", **workspace)
        client.close()


if __name__ == "__main__":
    main()
