"""Geometry/energy/momentum math oracle and source contracts, not a GPU test."""
from collections import deque
from pathlib import Path
import math
import random
import struct

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'RayTrophiStudio/source'
OFFSETS = ((0,1,0),(-1,0,0),(1,0,0),(0,0,-1),(0,0,1),(0,-1,0))


def faces(cell, occupied, solid=frozenset(), size=8, periodic=False):
    result = []
    for direction in OFFSETS:
        neighbour = tuple(a+b for a,b in zip(cell,direction))
        if periodic:
            neighbour = tuple(a % size for a in neighbour)
        if all(0 <= a < size for a in neighbour) and neighbour not in occupied and neighbour not in solid:
            result.append(neighbour)
    return result


def nearest(occupied, solid=frozenset(), fluid=None):
    fluid = occupied if fluid is None else fluid
    release, queue = {}, deque()
    for cell in sorted(fluid):
        available = faces(cell, occupied, solid)
        if available:
            release[cell] = available[0]
            queue.append(cell)
    while queue:
        cell = queue.popleft()
        for direction in OFFSETS:
            neighbour = tuple(a+b for a,b in zip(cell,direction))
            if neighbour in fluid and neighbour not in release:
                release[neighbour] = release[cell]
                queue.append(neighbour)
    return release


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def main():
    isolated = {(3,3,3)}
    assert len(faces((3,3,3), isolated)) == 6
    jet = {(3,y,3) for y in range(1,7)}
    assert len(faces((3,3,3), jet)) == 4
    assert nearest(jet)[(3,3,3)][1] == 3  # local side, not top of the entire jet
    floor = {(x,0,z) for x in range(8) for z in range(8)}
    sheet = {(x,1,z) for x in range(2,5) for z in range(2,5)}
    assert sum(len(faces(c, sheet, floor)) for c in sheet) == 21
    cube = {(x,y,z) for x in range(2,5) for y in range(2,5) for z in range(2,5)}
    assert (3,3,3) in nearest(cube)
    all_cells = {(x,y,z) for x in range(8) for y in range(8) for z in range(8)}
    assert not nearest(cube, all_cells-cube)
    assert (7,3,3) in faces((0,3,3), {(0,3,3)}, periodic=True)

    rng = random.Random(39)
    for count in range(1,7):
        for _ in range(100):
            mass = 10 ** rng.uniform(-9,2)
            specific, latent = 4184., 2257000.
            temperature = rng.uniform(280,500)
            cp = mass * specific
            remaining_energy = vapour_energy = heat = evaporated = remaining_cp = 0.
            for _ in range(count):
                piece_cp = cp/count
                piece_mass = mass/count
                air = rng.uniform(240,550)
                ca = 10 ** rng.uniform(-6,3)
                relax = -math.expm1(-.1 * (1/piece_cp + 1/ca))
                q = (temperature-air) * piece_cp * ca/(piece_cp+ca) * relax
                after_heat = temperature - q/piece_cp
                budget = piece_cp * max(0,after_heat-1)/latent
                dm = min(.5*piece_mass*rng.random(), budget)
                final = after_heat - dm*latent/piece_cp
                assert final >= 1 - 1e-8
                capacity = piece_cp - dm*specific
                remaining_energy += capacity * final
                remaining_cp += capacity
                vapour_energy += dm * (specific*final+latent)
                heat += q
                evaporated += dm
            assert evaporated <= .5*mass*(1+1e-12)
            assert abs(remaining_energy+vapour_energy+heat-cp*temperature) <= max(1e-8,cp*temperature*1e-10)
            mixed_temperature = remaining_energy/remaining_cp
            assert mixed_temperature > 0 and math.isfinite(mixed_temperature)
    # A complete boil retains superheat beyond the finite latent requirement.
    for initial in (373.15001,435,2000):
        mass, specific, boil, latent = .0149, 4184., 373.15, 2257000.
        superheat = mass*specific*(initial-boil)
        dm = min(mass,superheat/latent)
        gas = dm*specific*boil + max(0,superheat-dm*latent) + dm*latent
        assert abs((mass-dm)*specific*boil+gas-mass*specific*initial) < 1e-8
    # Sparse MAC mixing uses the physical transferred mass, not a velocity kick.
    for _ in range(500):
        before, added = 10**rng.uniform(-7,2),10**rng.uniform(-8,1)
        old, source = rng.uniform(-20,20),rng.uniform(-20,20)
        velocity = (before*old+added*source)/(before+added)
        assert min(old,source)-1e-10 <= velocity <= max(old,source)+1e-10
        assert abs((before+added)*velocity-before*old-added*source) <= 1e-9*max(1,before+added)

    geometry = (SRC/'src/Physics/Fluid/GasSurfaceGeometry.cpp').read_text()
    assert 'liquid_cells_.find(next)' in geometry and 'std::sort(cells.begin()' in geometry
    plan = (SRC/'src/Physics/Fluid/GasSurfaceExchangePlan.cpp').read_text()
    assert '1.0f / faces.count' in plan and 'capacity * split' in plan
    assert 'std::stable_sort(plan.records' in plan and 'queueGasPhaseMomentum' in plan
    # Rain absorbed by a parcel joins its capacity like condensate (part 3).
    assert 'input.capacity - (value.mass - rain) * input.specific_heat' in plan
    assert 'mix.evaporated += value.mass - rain' in plan
    assert 'MatterExchangeKind::Precipitation' in plan
    cpu = (SRC/'src/Physics/Fluid/GasSurfaceExchangeCpu.cpp').read_text()
    shader = (SRC/'shaders/sim_inventory_surface_exchange.comp').read_text()
    assert 'tp -= dm * latent / cp' in cpu and 'tp -= dm * latent / cp' in shader
    assert 'cp - dm * c' not in cpu and 'cp - dm * c' not in shader
    boiling = (SRC/'src/Physics/Fluid/FluidMistPhaseExchange.cpp').read_text(encoding='utf-8')
    assert 'surface.nearestAir' in boiling and 's_column_top' not in boiling
    assert 'surplus_j' in boiling and 'std::nextafter' in boiling
    momentum = (SRC/'src/Physics/Fluid/GasPhaseMomentum.cpp').read_text()
    assert 'std::map<uint64_t, FaceDeposit>' in momentum and 'markDeviceWrote' in momentum
    assert 'record.momentum) / (before + record.mass)' in momentum
    print('PASS six-face/local-vent/energy/momentum oracle and shared source contracts; no GPU execution')


if __name__ == '__main__':
    main()
