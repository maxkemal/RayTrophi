# Viewport measurement smoke test.
#
# This is the loop the agent could NOT close on 2026-08-16: drive nothing, read
# nothing, and answer "is there a black band?" by eye. Every number below was a
# human copying a panel by hand that day.
#
# Run:  rt.script.run_file("scripts/test/rt_test_viewport_probe.py")
import rt

FAIL = []


def check(label, ok, detail=""):
    print(("  OK   " if ok else "  FAIL ") + label + ((" -- " + detail) if detail else ""))
    if not ok:
        FAIL.append(label)


print("== viewport.status (capture off) ==")
st = rt.viewport.status()
print("   backend=%s %dx%d samples=%d active=%s complete=%s" % (
    st["backend"], st["width"], st["height"], st["samples"],
    st["rendering_active"], st["accumulation_complete"]))
check("backend bound", st["available"], "no backend => every counter below is meaningless")

# ★ An idle viewport reports zeros that look exactly like a cheap scene. This is
# the distinction that cost a diagnosis round, so assert it explicitly rather
# than trusting a zero.
if st["available"] and st["samples"] == 0 and not st["rendering_active"]:
    print("   NOTE: viewport idle and no samples -- counters are ABSENT, not zero")

print("== probe before capture is enabled ==")
p = rt.render.probe()
check("probe reports unavailable when capture is off", not p["available"],
      "an available=True here would mean a stale frame is being described")

print("== enable capture ==")
rt.viewport.capture(True)
st = rt.viewport.status()
check("capture_enabled reflects the request", st["capture_enabled"])

print("== probe after a frame is displayed ==")
p = rt.render.probe()
if not p["available"]:
    print("   frame not captured yet -- let the viewport draw one frame, then re-run")
else:
    print("   %dx%d px=%d mean=%.4f min=%.4f max=%.4f black=%.4f nan=%.4f" % (
        p["width"], p["height"], p["pixels"], p["mean_luminance"],
        p["min_luminance"], p["max_luminance"], p["black_fraction"], p["nan_fraction"]))
    print("   histogram=%s" % (p["histogram"],))
    check("region covers the frame", p["width"] == st["width"] and p["height"] == st["height"])
    check("fractions are in range",
          0.0 <= p["black_fraction"] <= 1.0 and 0.0 <= p["nan_fraction"] <= 1.0)
    # ★ A NaN is neither black nor lit and disappears inside a mean. Any nonzero
    # value here is a real defect, not a tolerance to tune.
    check("no NaN pixels", p["nan_fraction"] == 0.0,
          "nan_fraction=%.6f" % p["nan_fraction"])

    print("== sub-region probe (centre quarter) ==")
    q = rt.render.probe(x=p["width"] // 4, y=p["height"] // 4,
                        width=p["width"] // 2, height=p["height"] // 2)
    check("sub-region is smaller than the frame",
          q["available"] and q["pixels"] < p["pixels"],
          "%d vs %d" % (q.get("pixels", -1), p["pixels"]))

print("== volume counters cross-check ==")
v = rt.render.volume_stats()
if not v["available"]:
    print("   no Vulkan backend -- skipped")
elif not v["enabled"]:
    print("   counters disabled -- all zeros below would be ABSENCE, not measurement")
elif v["volume_rays"] == 0:
    print("   volume_rays=0 -- viewport idle or no volume in view, NOT a cheap scene")
else:
    ratio = float(v["density_samples"]) / float(v["volume_rays"])
    print("   density/ray = %.4f  (>=0.5 healthy, <=0.02 = rays enter and do no work)" % ratio)
    check("volume rays do real work", ratio > 0.02,
          "density/ray=%.4f suggests volume-box re-entry" % ratio)

rt.viewport.capture(False)
check("capture releases its buffer", not rt.viewport.status()["frame_available"])

print("")
print("RESULT: " + ("ALL PASSED" if not FAIL else "%d FAILED: %s" % (len(FAIL), FAIL)))
