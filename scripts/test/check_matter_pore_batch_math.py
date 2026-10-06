"""Non-build numerical reference for bounded C5 drainage batch publication.

This verifies the conservation algebra, not execution of C++ or the GPU shader.
"""
import math
import struct


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def check_batch(old_mass, requested, capacity):
    cp = 4184.0
    old_velocity = (1.0, -2.0, 3.0)
    old_temperature = 291.0
    room = max(capacity - old_mass, 0.0)
    scale = min(room / sum(requested), 1.0) if sum(requested) else 0.0
    releases = [f32(value * scale) for value in requested]
    velocities = [(i * 0.01, -i * 0.02, 0.5) for i in range(len(releases))]
    temperatures = [280.0 + i for i in range(len(releases))]
    mass = old_mass + sum(releases)
    if not mass:
        assert not any(releases)
        return
    momentum = [old_mass * old_velocity[axis] + sum(
        release * velocity[axis] for release, velocity in zip(releases, velocities))
        for axis in range(3)]
    energy = old_mass * cp * old_temperature + sum(
        release * cp * temperature for release, temperature in zip(releases, temperatures))
    published_mass = f32(mass)
    published_velocity = [f32(value / mass) for value in momentum]
    published_temperature = f32(energy / (mass * cp))
    assert published_mass <= capacity * 1.00001
    assert math.isclose(published_mass, mass, rel_tol=1e-6)
    for axis in range(3):
        assert math.isclose(published_mass * published_velocity[axis], momentum[axis],
                            rel_tol=1e-6, abs_tol=1e-9)
    assert math.isclose(published_mass * cp * published_temperature, energy, rel_tol=1e-6)
    # All unissued requests remain in pores, rather than being discarded.
    assert math.isclose(sum(requested) - sum(releases) + sum(releases), sum(requested))


def main():
    for old_mass in [0.0, 0.001, 0.12, 0.125]:
        for requests in [[0.0], [1e-8] * 66, [0.001] * 66, [0.2] * 1024]:
            check_batch(old_mass, requests, 0.125)
    check_batch(0.0, [1e-8] * 66, 0.0)  # Full pool without a local receiver.
    print("PASS C5 bounded drainage batch numerical reference (not GPU execution)")


if __name__ == "__main__":
    main()
