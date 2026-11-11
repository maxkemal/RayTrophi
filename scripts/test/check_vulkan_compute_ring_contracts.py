"""Static contract: the Vulkan simulation backend's two-slot submission ring.

Every 512 dispatches the recording is submitted WITHOUT waiting and the other
slot records on (2026-11-11; snow + water ran host-bound because a full
submit+wait every 512 dispatches serialised host recording and GPU work).
The invariants below are what keeps that safe: a submission still in flight is
queued work like an open recording.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "RayTrophiStudio" / "source" / "src" / "Device" / "SimulationComputeVulkan.cpp"


def main():
    s = SRC.read_text(encoding="utf-8")
    # Two slots, at most one in flight.
    assert "static constexpr int kSlots = 2;" in s
    assert "void submitRecordingAsync()" in s and "void waitInFlight()" in s
    body = s[s.index("void submitRecordingAsync()"):s.index("void waitInFlight()")]
    assert body.index("waitInFlight();") < body.index("vkQueueSubmit("), "submit before freeing the other slot"
    assert "vkWaitForFences" not in body, "the async submit must not wait"
    assert "m_stagingUpCursor" not in body and "m_pendingDownloads" not in body, \
        "staging/downloads are released only by the full synchronize"
    # The 512 boundary: async, except timestamps and an open transfer batch.
    assert "if (m_timestampsEnabled || m_batchActive) synchronize();\n            else submitRecordingAsync();" in s
    # synchronize waits both slots.
    sync = s[s.index("void synchronize() override {"):s.index("// ── Dispatch")]
    assert "if (!m_recording && !m_inFlight) return;" in sync
    assert sync.count("waitInFlight();") == 2
    # A submission in flight blocks direct ReBAR writes and forces the recorded upload path.
    assert "if (it->second.mapped && !m_recording && !m_inFlight) {" in s
    assert "if (m_batchActive || m_recording || m_inFlight || m_stagingUpCursor > 0) {" in s
    # Lifetime boundaries drain it.
    assert s.count("if (m_recording || m_inFlight) synchronize();") == 3
    # The immediate copy path drains it; a new recording after it starts with a full barrier.
    copy = s[s.index("void submitCopyImmediate("):s.index("bool beginKernelTimestamp(")]
    assert "waitInFlight();" in copy
    rec = s[s.index("bool ensureRecording() {"):]
    assert rec.index("if (m_inFlight) {") < rec.index("vkCmdPipelineBarrier(")
    assert "VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT" in rec
    # Early submit was tried and removed (2026-11-11): v1 (idle polling) 83 -> 112 ms
    # on the test scene, v2 (first batch after a sync) no gain in the user scene --
    # the host/GPU gap there is serial CPU work at the frame edges, not post-sync bubbles.
    assert "kEarlySubmitDispatches" not in s and "m_drainedBySync" not in s
    print("PASS Vulkan compute two-slot ring: async 512 boundary, both-slot sync, ReBAR/destroy/copy guards, entry barrier")


if __name__ == "__main__":
    main()
