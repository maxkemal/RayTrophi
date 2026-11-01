"""Guard against benchmarks forcing CPU progressive passes or timing atomics."""
import unittest
from unittest.mock import patch
from pathlib import Path
from gas_probe_sampling import measure


class Client:
    def __init__(self):
        self.counters = False
        self.timing = False
        self.timed_reads = 0

    def call(self, method, **params):
        if method == "timeline.get_frame": return 49
        if method == "render.volume_counters": self.counters = params["enabled"]; return {}
        if method == "render.set_gpu_timing": self.timing = params["enabled"]; return {}
        if method == "reset_accumulation": return {}
        if method == "render.gpu_timing_stats":
            assert self.timing and not self.counters
            self.timed_reads += 1
            return {"samples": [{"gpu_ms": float(i)} for i in range(12)]}
        if method == "viewport.status":
            assert self.counters and not self.timing
            return {"samples": 4}
        if method == "render.volume_stats": return {"paths_traced": 100}
        raise AssertionError("Unexpected/forced render call: " + method)


class SamplingContracts(unittest.TestCase):
    def test_timing_discards_warmup_and_is_separate_from_counters(self):
        c = Client()
        timing, stats = measure(c, samples=4, warmup=8)
        self.assertEqual(timing["gpu_mean_ms"], 9.5)
        self.assertEqual(stats["paths_traced"], 100)
        self.assertFalse(c.timing)
        self.assertFalse(c.counters)

    def test_probes_do_not_force_progressive_cpu_passes(self):
        root = Path(__file__).parent
        for name in ("rt_probe_gas_max_steps_ipc.py", "rt_probe_gas_overlap_cost_ipc.py"):
            source = (root / name).read_text(encoding="utf-8")
            self.assertNotIn('call("viewport.render_frames"', source)
            self.assertIn("from gas_probe_sampling import measure", source)


if __name__ == "__main__":
    unittest.main()
