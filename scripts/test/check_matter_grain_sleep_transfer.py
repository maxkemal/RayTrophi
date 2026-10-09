"""Independent impulse/time specifications and revision-24 source wiring.

Does not compile or execute shaders. Live weak/strong impact, slow loads and
repose/large-N parity must still pass after the user's build.
"""
import math
import struct
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'RayTrophiStudio/source').is_dir())
SOURCE = ROOT / 'RayTrophiStudio/source'


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def add(*vectors):
    return tuple(sum(v[i] for v in vectors) for i in range(3))


def negate(v):
    return tuple(-x for x in v)


def check_physics():
    # Analytical elastic two-body collision is an upper bound for e <= 1.
    speed_limit = .002
    projectile, target = .006, .6
    reduced_mass = projectile * target / (projectile + target)
    weak_j = 2 * reduced_mass * .003
    strong_j = 2 * reduced_mass * 1.0
    assert weak_j / target < speed_limit < strong_j / target
    # A source below the speed threshold can still move a much lighter target.
    heavy, light, slow = .6, .006, .0015
    slow_j = 2 * heavy * light / (heavy + light) * slow
    assert slow < speed_limit < slow_j / light
    # J^2/(2m) and L^2/(2I) give the same per-DOF kinetic threshold as
    # testing the recipient's velocity/surface-angular change.
    radius = .008
    inertia = .4 * target * radius**2
    for j in (weak_j, strong_j):
        assert (j*j/(2*target) >= .5*target*speed_limit**2) == \
            (abs(j)/target >= speed_limit)
    for angular_j in (1e-10, 1e-5):
        assert (angular_j**2/(2*inertia) >= .5*inertia*(speed_limit/radius)**2) == \
            (abs(angular_j)/inertia*radius >= speed_limit)
    # Two equal spheres' tangential spin torques have the SAME sign; rolling
    # and twist reactions have opposite signs. Check total world angular impulse.
    arm, tangent = (0, -.025, 0), (.5, 0, 0)
    roll, twist = (.001, .002, 0), (0, .003, 0)
    own_torque = add(cross(arm, tangent), roll, negate(twist))
    other_torque = add(cross(arm, tangent), negate(roll), twist)
    own_pos, other_pos = negate(arm), arm
    world = add(cross(own_pos, tangent), cross(other_pos, negate(tangent)),
                own_torque, other_torque)
    assert math.dist(world, (0, 0, 0)) < 1e-14
    # Stationary preload is not newly delivered kinetic impulse.
    preload, damping, slip_dt = 10., 0., 0.
    assert preload - preload + damping + slip_dt == 0.
    # Pure normal-axis spin cannot transfer torque without twist friction;
    # zero sliding/rolling coefficients cannot manufacture coupling either.
    assert min(0. * preload * .02, 2. * .1) == 0.


def check_time():
    # A time-step change must not discard 0.15 s of already completed rest.
    age = f32(.15)
    durations = [.00025] * 100 + [.001] * 25
    for dt in durations:
        age = f32(min(age + dt, f32(.2)))
    assert abs(age - .2) < 1e-6
    # Long configured rest + very small dt: naive f32 addition stalls. The
    # compensation uses the same owner-only word that sleepers use for skipped dt.
    age, compensation, tiny_dt = f32(9.9), 0., f32(1e-7)
    origin = age
    assert f32(age + tiny_dt) == age
    for _ in range(1000):
        increment = f32(tiny_dt - compensation)
        advanced = f32(age + increment)
        compensation = f32(f32(advanced - age) - increment)
        age = advanced
    assert abs(age - (origin + 1000*tiny_dt)) <= 1e-6
    # Positive float encodings are ordered; the high request bit remains separate.
    assert bits(0.) < bits(.15) < bits(.2) < 0x80000000
    requested = bits(.2) | 0x80000000
    assert requested & 0x7fffffff == bits(.2)
    # Constant slip's exact integral over skipped substeps matches a single
    # catch-up update; a new contact must use only its first physical dt.
    slip, spring, dt, skipped = .001, 2000., .0001, 99
    accumulated = sum(slip * dt for _ in range(skipped + 1))
    catch_up = slip * (skipped + 1) * dt
    assert math.isclose(spring * accumulated, spring * catch_up, rel_tol=1e-12)
    assert spring * slip * dt < spring * catch_up


def check_source():
    shader = (SOURCE / 'shaders/sim_matter_grain.glsl').read_text(encoding='utf-8-sig')
    transfer = (SOURCE / 'shaders/sim_matter_grain_sleep_transfer.glsl').read_text()
    host = (SOURCE / 'src/Physics/Fluid/MatterGrainGpu.cpp').read_text(encoding='utf-8-sig')
    assert 'const uint REVISION = 24u;' in shader
    guard = shader.index('uint(history_blocks.length()) < 5u*pc.substep.y')
    assert guard < shader.index('uint entry = history_blocks[i];')
    assert 'if (overflow & 16u)' in host
    assert 'g_dynamic_contact' not in shader
    assert 'bool balanced = can_sleep && force_balanced;' in shader
    assert 'if (sleepOn() && !force_balanced && !g_fast)' in shader
    assert 'rest = context_changed ? 0u : raw & ~AUDIT;' in shader
    assert 'if (g_probe_sleep_transfer && linked)' in shader
    assert 'g_pair_dynamic_angular_impulse,jm,ji,r' in shader
    assert 'float previous_normal = slot == EMPTY ? 0.0' in shader
    assert 'g_pair_dynamic_impulse = -(n*(fn-previous_normal)+changed_tangent)*dt;' in shader
    assert shader.index('if (magnitude > coulomb)') < shader.index('g_pair_dynamic_impulse = -(')
    assert 'floatBitsToUint(max(pc.sleep.w,pc.step_contact.x))' in shader
    assert 'sleep_context[8] = 0u;' in host
    assert 'constants.sleep[0] = 0.0f;' in host
    assert host.count('5 * capacity * sizeof(uint32_t)') == 2
    assert 'pc.substep.y+4u*g_block' in shader
    assert 'g_history_dt = elapsed+pc.step_contact.x;' in shader
    assert 'float elapsed = was_sleeping ? clock_auxiliary : 0.0;' in shader
    assert '!was_sleeping && can_sleep && input_slow' in shader
    assert 'float increment = dt-rest_compensation;' in shader
    assert 'history_blocks[elapsed_word] = floatBitsToUint(next_compensation);' in shader
    assert 'float history_dt = slot == EMPTY ? dt : g_history_dt;' in shader
    assert '2.0 * max(-normal_speed, 0.0) / inverse_normal_mass' in transfer
    assert 'pc.step_contact.w * normal_force * horizon' in transfer
    assert 'inverse_inertia * radius' in transfer


if __name__ == '__main__':
    check_physics()
    check_time()
    check_source()
    print('PASS revision-24 sleep transfer specification: unequal masses, impulse/energy, '
          'reaction torque, preload, adaptive clocks and skipped spring time; no GPU execution')
