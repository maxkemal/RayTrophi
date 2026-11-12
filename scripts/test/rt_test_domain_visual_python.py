"""Embedded API contract check. Contains no named-pipe client or waits."""
import rt


def run(domain, expected=None):
    before = rt.fluid.visual_samples(domain, limit=16)
    if expected is not None:
        assert before["items"] == expected, "Python and IPC see different visual samples"
    try:
        rt.fluid.set_visual_samples(domain, max_samples=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Python accepted a boolean sample count")
    after = rt.fluid.visual_samples(domain, limit=16)
    assert after["settings"] == before["settings"]
    assert after["items"] == before["items"], "Python read/validation changed sample state"
    print("PASS Python/IPC shared field-sample service")


if __name__ == "__main__":
    run("T_FieldSamples")
