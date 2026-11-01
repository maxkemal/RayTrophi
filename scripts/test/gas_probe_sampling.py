"""Measure natural Vulkan dispatches; collect instrumentation in a separate pass."""
import time


def measure(client, samples=4, warmup=8, timeout=55):
    frame = client.call("timeline.get_frame")
    count = max(1, samples)
    try:
        client.call("render.volume_counters", enabled=False)
        client.call("render.set_gpu_timing", enabled=True)
        client.call("reset_accumulation")
        deadline = time.monotonic() + timeout
        while True:
            if client.call("timeline.get_frame") != frame:
                raise RuntimeError("Timeline changed during GPU measurement")
            timing = client.call("render.gpu_timing_stats")
            captured = timing.get("samples", [])
            if len(captured) >= warmup + count:
                kept = captured[-count:]
                timing = dict(timing, samples=kept,
                    gpu_mean_ms=sum(x["gpu_ms"] for x in kept)/len(kept),
                    sampling_method="natural_gpu_counters_disabled")
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("Natural GPU timestamp progress timed out")
            time.sleep(.05)
        client.call("render.set_gpu_timing", enabled=False)
        client.call("render.volume_counters", enabled=True)
        client.call("reset_accumulation")
        deadline = time.monotonic() + timeout
        while True:
            if client.call("timeline.get_frame") != frame:
                raise RuntimeError("Timeline changed during counter measurement")
            status = client.call("viewport.status")
            if status.get("samples", 0) >= max(count, 4):
                counters = client.call("render.volume_stats")
                if counters.get("paths_traced", 0) > 0:
                    break
            if time.monotonic() >= deadline:
                raise RuntimeError("Natural volume counter progress timed out")
            time.sleep(.05)
        return timing, counters
    finally:
        # The caller restores the user's original instrumentation state.
        client.call("render.set_gpu_timing", enabled=False)
        client.call("render.volume_counters", enabled=False)
