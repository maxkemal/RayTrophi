"""CPU math oracle for the film approximation and 2D dispatch, not GPU execution."""
import math
import struct


def f32(x):
    return struct.unpack('f', struct.pack('f', x))[0]


def relax(x):
    if x < .05:
        return x * (1 - x * (.5 - x * (1 / 6 - x * (1 / 24 - x * (1 / 120 - x / 720)))))
    return f32(1 - f32(math.exp(f32(-x))))


def main():
    for i in range(-14, 7):
        for factor in (1, 2, 5):
            x = factor * 10.0 ** i
            actual, expected = relax(x), -math.expm1(-x)
            assert abs(actual - expected) <= max(2e-7, expected * 2e-6), (x, actual, expected)
            assert 0 <= actual <= 1
            for cp in (1e-8, 1, 1e8):
                for ca in (1e-8, 1, 1e8):
                    tp, ta = 500., 290.
                    q = (tp - ta) * (cp * ca / (cp + ca)) * actual
                    p, a = tp - q / cp, ta + q / ca
                    assert ta - 1e-8 <= p <= tp + 1e-8
                    assert ta - 1e-8 <= a <= tp + 1e-8
                    assert abs((p - tp) * cp + (a - ta) * ca) <= max(1e-5, abs(q) * 1e-7)
    for threads in (128, 256):
        for cells in (1, 255, 256, 257, 128 ** 3, 256 ** 3, 2 ** 31 - 1):
            blocks = (cells + threads - 1) // threads
            gx = min(65535, max(1, blocks))
            gy = (blocks + gx - 1) // gx
            assert gx * gy >= blocks and (gy - 1) * gx < blocks
            assert gx * gy * threads < 2 ** 32
            # Shader block id must enumerate each valid reduction slot once.
            for block in (0, min(blocks - 1, gx - 1), blocks - 1):
                y, x = divmod(block, gx)
                assert x + y * gx == block and y < gy
    # A thin impulse can disappear from every semi-Lagrangian backtrace.
    # Closed/periodic fields must survive a zero denominator; open outflow stays zero.
    previous = [0., .128675996, 0., 0.]
    for closed in (False, True):
        result = [0.] * len(previous)
        before, after = sum(previous), sum(result)
        if closed and before > 0 and after <= 0:
            result = previous[:]
        assert sum(result) == (before if closed else 0)
    from pathlib import Path
    source = Path(__file__).resolve().parents[2] / 'RayTrophiStudio/source'
    shader = (source / 'shaders/sim_inventory_conserve.comp').read_text()
    assert 'totals.x > 0.0 && totals.y <= 0.0' in shader
    assert 'totals.z > 0.0 && totals.w <= 0.0' in shader
    cpu = (source / 'src/Physics/Fluid/FluidMistPhaseExchange.cpp').read_text(encoding='utf-8')
    assert 'preserve_total && !(after > 0.0)' in cpu and 'field = previous' in cpu
    assert 'fluid_state.thermal_stats.evaporated_kg += boiled_kg' in cpu
    print('PASS film/dispatch/zero-advection conservation oracle and source contracts; no GPU execution')


if __name__ == '__main__':
    main()
