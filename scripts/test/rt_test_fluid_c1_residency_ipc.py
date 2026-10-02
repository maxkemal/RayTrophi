"""C1 live gate: Vulkan G2P and advect-tail share one particle readback."""

import json
import uuid

from rt_ipc import RtIpc


MAX_DOWNLOAD_BYTES = 9_430_584
MAX_BATCH_ENDS = 12


def run(client, domain):
    for _ in range(4):
        client.call("fluid.step", dt=1.0 / 60.0)
    return (
        client.call("fluid.state_digest", domain=domain),
        client.call("fluid.step_stats", domain=domain),
    )


def main():
    client = RtIpc()
    domain = "C1Residency_" + uuid.uuid4().hex[:8]
    created = False
    try:
        assert not client.call("fluid.list_domains")["domains"], (
            "C1 acceptance needs an otherwise empty scene")
        client.call(
            "fluid.create_domain", name=domain, type="fluid",
            domain_min=[0.0, 0.0, 0.0], domain_max=[2.0, 2.0, 2.0],
            voxel_size=0.05)
        created = True
        client.call(
            "fluid.set_param", domain=domain, backend="vulkan",
            boundary="closed", visible=False, max_particles=100000)
        client.call(
            "fluid.seed", domain=domain,
            seed_min=[0.025, 0.025, 0.025],
            seed_max=[1.975, 1.975, 1.975],
            particles_per_cell=2, persistent=True)

        digest_a, stats_a = run(client, domain)
        client.call("fluid.reset")
        digest_b, stats_b = run(client, domain)
        centroid_error = max(
            abs(float(a) - float(b))
            for a, b in zip(digest_a["centroid"], digest_b["centroid"])
        )
        mean_speed_error = abs(
            float(digest_a["mean_speed"]) - float(digest_b["mean_speed"])
        )
        report = {
            "digest_bit_exact": digest_a == digest_b,
            "centroid_error": centroid_error,
            "mean_speed_error": mean_speed_error,
            "particle_count": stats_b.get("particle_count"),
            "g2p_on_gpu": stats_b.get("g2p_on_gpu"),
            "download_bytes": stats_b.get("download_bytes"),
            "batch_end_calls": stats_b.get("batch_end_calls"),
            "total_ms": stats_b.get("total_ms"),
        }
        print(json.dumps(report, indent=2))
        assert digest_a["particles"] == digest_b["particles"] == 100000
        assert centroid_error <= 1.0e-6
        assert mean_speed_error <= 1.0e-6
        assert report["particle_count"] == 100000
        assert report["g2p_on_gpu"] is True
        assert report["download_bytes"] <= MAX_DOWNLOAD_BYTES
        assert report["batch_end_calls"] <= MAX_BATCH_ENDS
        print("PASS: C1 combined G2P + advect-tail readback")
    finally:
        if created:
            client.try_call("fluid.remove_domain", domain=domain)
        client.close()


if __name__ == "__main__":
    main()
