"""Source-only DEM sleep audit plus independent physical/race specification checks.

No shader compilation or GPU execution. Actual sleep/diagnostic parity is tested
by rt_test_grain_sleep_ipc.py after the user's revision-24 build.
"""

import itertools
import math
import re
import struct
from pathlib import Path

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / 'RayTrophiStudio/source').is_dir())
SOURCE = ROOT / 'RayTrophiStudio/source'
AUDIT = 0x80000000


def read(path):
    return (SOURCE / path).read_text(encoding='utf-8-sig')


def check_source():
    shader = read('shaders/sim_matter_grain.glsl')
    policy = read('shaders/sim_matter_grain_sleep.glsl')
    host = read('src/Physics/Fluid/MatterGrainGpu.cpp')
    assert 'const uint REVISION = 24u;' in shader
    assert '#include "sim_matter_grain_sleep.glsl"' in shader
    assert 'pc.substep.x == pc.substep.w || (pc.substep.x + 1u) % period == 0u' in policy
    assert 'float interval = min(0.02, pc.sleep.w);' in policy
    assert 'float horizon = max(pc.sleep.w, pc.step_contact.x);' in policy
    assert 'float tolerance = pc.sleep.y / horizon;' in policy
    assert 'linear_tolerance = min(linear_tolerance, 0.25 * gravity);' in policy
    assert 'dot(acceleration, acceleration) < linear_tolerance * linear_tolerance' in policy
    assert '2.0 * ulp * pc.high_stiffness.w * inverse_mass * float(contacts)' in policy
    assert 'dot(angular_acceleration, angular_acceleration) * radius * radius' in policy
    assert 'atomicCompSwap(history_blocks[word], previous, desired)' in policy
    assert 'uint desired = (previous & AUDIT) | rest;' in policy
    assert 'previous = observed;' in policy
    assert 'atomicAnd(history_blocks[rest_word], AUDIT)' not in shader
    assert 'history_blocks[jw] & ~AUDIT' not in shader
    assert 'coherent buffer HistoryBlocks' in shader
    assert 'if ((history_blocks[word] & ~AUDIT) >= sleepRestLimit())' in shader
    assert 'atomicOr(history_blocks[word], AUDIT);' in shader
    assert 'grainSleepTakeAudit(rest_word)' in shader
    assert 'history_blocks[elapsed_word] = floatBitsToUint(g_history_dt);' in shader
    assert 'float history_dt = slot == EMPTY ? dt : g_history_dt;' in shader
    assert 'history_blocks[elapsed_word] = 0u;' in shader
    assert 'previous_tangent+slip*history_dt' in shader
    assert 'previous_roll+roll*history_dt' in shader
    assert 'if ((previous & AUDIT) != 0u)' in policy
    assert 'if (history_blocks[word] != 0u)' in policy
    assert 'sleepOn() && valid_block && can_sleep && input_slow' in shader
    # Coupling and externally injected motion are checked before any sleep skip.
    assert shader.index('vec4 drag = coupling[3u*i];') < shader.index('if (sleeping && !audit_requested && !grainSleepAudit())')
    assert 'rest = context_changed ? 0u : raw & ~AUDIT;' in shader
    assert 'bool linked = g_bridges != bridges_before;' in shader
    assert 'if (g_probe_sleep_transfer && linked)' in shader
    assert re.search(r'bool force_balanced = sleepOn\(\) &&\s*grainSleepBalanced\(', shader)
    assert 'bool balanced = can_sleep && force_balanced;' in shader
    assert 'g_dynamic_contact' not in shader
    audited_return = shader.index('if (sleeping && balanced)')
    for word in (2, 3, 4, 5):
        assert shader.index(f'diagnostics[{word}],g_') < audited_return
    assert 'if (sleepOn() && !force_balanced && !g_fast)' in shader
    assert 'bool still = sleepOn() && balanced &&' in shader
    assert 'constants.sleep[3] = params.sleep_time_s;' in host
    assert 'sleep_context[8] = 0u;' in host
    assert 'constants.sleep[0] = 0.0f;' in host
    assert 'next_rest = floatBitsToUint(min(advanced_time,target_time));' in shader
    assert 'runtime.sleep_context != sleep_context' in host
    assert 'runtime.sleep_collider_fingerprint != runtime.collider_fingerprint' in host
    assert host.index('p.position = std::move(new_position);') < host.index('runtime.sleep_context = sleep_context;')
    # No added GPU storage/dispatch or changed push/descriptor ABI.
    assert len(re.findall(r'binding\s*=\s*(\d+)', shader)) == 17
    assert 'static_assert(sizeof(Constants) == 128);' in host
    assert '5 * capacity * sizeof(uint32_t)' in host


def check_physical_specification():
    speed, duration, radius = .002, .2, .025

    def balanced(acceleration, angular_acceleration):
        tolerance = min(speed / duration, .25 * 9.81)
        return math.dist(acceleration, (0, 0, 0)) < tolerance and \
            math.dist(angular_acceleration, (0, 0, 0)) * radius < tolerance

    assert balanced((0, 0, 0), (0, 0, 0))  # equilibrium may sleep.
    assert balanced((.001, 0, 0), (0, .01, 0))  # tolerate small residuals.
    assert not balanced((0, -9.81, 0), (0, 0, 0))  # zero-speed free fall is not rest.
    assert not balanced((.1, 0, 0), (0, 0, 0))  # lost support / slow load.
    assert not balanced((0, 0, 0), (0, 1, 0))  # rotational turning point.
    # In the aggressive live fixture, speed-only sleeping would freeze a fall
    # after .01 s. Even its 1 m/s threshold does not allow gravity equilibrium.
    assert 9.81 * .01 < 1.0
    assert 9.81 > min(1.0 / .01, .25 * 9.81)  # force guard rejects the aggressive fixture.
    # Independent next-representable-float oracle for the shader's position ULP.
    # An 8 mm, 5.7 g grain at y=2 m has a quantized spring-force residual far
    # above .01 m/s². Rejecting all of that noise would disable large-N sleep.
    for coordinate in (.008, .025, .233, 1., 2., 10., 1000.):
        bits = struct.unpack('<I', struct.pack('<f', coordinate))[0]
        actual = struct.unpack('<f', struct.pack('<I', bits))[0]
        following = struct.unpack('<f', struct.pack('<I', bits + 1))[0]
        exponent = (bits >> 23) & 255
        shader_ulp = struct.unpack('<f', struct.pack('<I', (exponent - 23) << 23))[0]
        assert shader_ulp == following - actual
    mass = 1600 / .6 * 4 / 3 * math.pi * .008**3
    uncertainty = 2 * 2**-22 * 20000 / mass * 4
    quantized_tolerance = min(max(speed / duration, uncertainty), .25 * 9.81)
    assert speed / duration < .5 < quantized_tolerance < 9.81
    for dt, steps in ((1/60/384, 384), (1/120/128, 128), (.001, 100), (.05, 2)):
        period = max(1, math.floor(min(.02, duration) / dt))
        audits = [i for i in range(steps) if i == steps - 1 or (i + 1) % period == 0]
        assert audits[-1] == steps - 1
        previous = -1
        for i in audits:
            assert (i - previous) * dt <= max(.02, dt) + 1e-12
            previous = i
        if steps >= 128:
            assert len(audits) < steps // 20  # preserve the large-substep fast path.


def check_atomic_interleavings():
    # Former failure: the owner exposes rest=0 between two writes, so a fast
    # neighbour peeks at it, decides it is awake, and never sets AUDIT.
    rest = 100
    word = rest & AUDIT
    neighbour_wakes = (word & ~AUDIT) >= rest
    word |= rest
    assert not neighbour_wakes and word == rest
    # Positive rest commit: an eligible neighbour OR and owner load/CAS (retry
    # after failure). Enumerate every interleaving that preserves owner order.
    for previous_rest, next_rest in itertools.product((0, 1, 100, 10**9), (1, 100, 10**9)):
        for order in (('load', 'cas', 'wake'), ('load', 'wake', 'cas'), ('wake', 'load', 'cas')):
            word, expected = previous_rest, None
            for operation in order:
                if operation == 'wake':
                    word |= AUDIT
                elif operation == 'load':
                    expected = word
                else:
                    # One failed CAS observes the newly set bit; retry must
                    # derive its replacement from that observed value.
                    if word != expected:
                        expected = word
                    word = (expected & AUDIT) | next_rest
            assert word == (AUDIT | next_rest), (order, previous_rest, next_rest, word)

    # Coherent peek does not send wakes to awake candidates. Their full solve
    # qualifies motion/balance before they can sleep; sleeping owners keep the
    # counter stable across a positive commit, eliminating transient zero.
    threshold = 100
    for counter in (0, 1, 99, 100):
        word = counter
        if (word & ~AUDIT) >= threshold:
            word |= AUDIT
        assert bool(word & AUDIT) == (counter >= threshold)
    # A wake arriving after a zero peek may remain until the next dispatch;
    # a positive CAS preserves it. Zero commits already execute the full solve.
    word = 100
    snapshot = word
    word |= AUDIT
    word = (word & AUDIT) | snapshot
    assert word == AUDIT | 100
    word = 0  # A non-resting owner cannot skip contacts after dropping AUDIT.
    assert (word & ~AUDIT) < threshold


def main():
    check_source()
    check_physical_specification()
    check_atomic_interleavings()
    print('PASS DEM sleep source/specification: diagnostics audits, physical balance, '
          'bridge/context wake, CAS interleavings and unchanged 17/128 ABI')


if __name__ == '__main__':
    main()
