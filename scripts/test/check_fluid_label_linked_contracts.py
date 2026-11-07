"""Non-build linked membership/label specification and source contracts.

This does not execute SPIR-V. Live GPU/CPU/memory acceptance requires user build.
"""
import itertools
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'RayTrophiStudio/source'


def read(path):
    return (ROOT / path).read_text(encoding='utf-8')


def cell(point):
    return tuple(math.floor(axis) for axis in point)


def check_membership_and_labels():
    # >64 members in one bin, isolated points, a 4-neighbour hysteresis group,
    # negative coordinates and adjacent-bin neighbours. Unit radius is exact.
    points = [(0.125, 0.125, 0.125)] * 128
    points += [(-4.0 + i / 32, -3.0, -3.0) for i in range(5)]
    points += [(20.0, 0.0, 0.0), (22.0, 0.0, 0.0)]
    points += [(-10.0625, 0.0, 0.0), (-9.9375, 0.0, 0.0)]
    radius_squared = 1.0
    neighbours = []
    for own, p in enumerate(points):
        # Independent all-pairs geometric oracle: no bins or capacity limit.
        neighbours.append(min(6, sum(
            own != other and sum((a - b)**2 for a, b in zip(p, q)) <= radius_squared
            for other, q in enumerate(points))))
    rng = random.Random(20261106)
    for _ in range(24):
        order = list(range(len(points)))
        rng.shuffle(order)
        heads, links, delayed = {}, {}, []
        for index in order:
            key = cell(points[index])
            delayed.append((index, heads.get(key)))
            heads[key] = index
        # GPU atomic exchange and ordinary next writes need not complete in
        # matching order. The inter-dispatch barrier must complete all writes.
        rng.shuffle(delayed)
        links.update(delayed)
        membership = set()
        for key, head in heads.items():
            visited = set()
            while head is not None:
                assert head not in visited and cell(points[head]) == key
                visited.add(head)
                head = links[head]
            assert not membership.intersection(visited)
            membership.update(visited)
        assert membership == set(range(len(points)))
        for own, p in enumerate(points):
            found = 0
            center = cell(p)
            for offset in itertools.product((-1, 0, 1), repeat=3):
                key = tuple(a + b for a, b in zip(center, offset))
                head = heads.get(key)
                while head is not None:
                    if own != head and sum((a - b)**2 for a, b in zip(p, points[head])) <= 1.0:
                        found += 1
                    head = links[head]
            assert min(6, found) == neighbours[own]
            for previous in ('body', 'spray'):
                actual = 'spray' if found <= 2 else 'body' if found >= 6 else previous
                expected = 'spray' if neighbours[own] <= 2 else (
                    'body' if neighbours[own] >= 6 else previous)
                assert actual == expected


def check_source_contracts():
    host = read('src/Physics/Fluid/FluidParticleLabelsGpu.cpp')
    shader = read('shaders/sim_fluid_label_linked.glsl')
    backend = read('src/Device/SimulationComputeVulkan.cpp')
    assert 'const std::size_t item_bytes = count * sizeof(uint32_t);' in host
    assert 'kMaxPerBin' not in host
    assert 'gpu_stats[4] != 7u' in host
    assert host.index('uploadBuffer(buffers.label_stats, gpu_stats, stat_bytes)') < host.index(
        'cmd.kernel = "sim_fluid_label_link_clear"')
    assert host.index('gpu_stats[4] != 7u') < host.index('particles.flags.assign')
    assert 'capacity <= keep_limit' in host  # discard oversized retained scratch
    for stage, bindings, bit in [('clear', 2, 1), ('scatter', 5, 2), ('classify', 6, 4)]:
        name = 'sim_fluid_label_link_' + stage
        assert 'cmd.kernel = "' + name + '";' in host
        entry = next(line for line in backend.splitlines() if f'"{name}"' in line)
        assert ''.join(entry.split()) == (
            '{"' + name + '","' + name + '.spv",' + str(bindings) + ',80},')
        assert f'#define LABEL_LINK_{stage.upper()}' in read('shaders/' + name + '.comp')
        assert ('stats[4] = 1u' if bit == 1 else f'atomicOr(stats[4], {bit}u)') in shader
    assert 'next_index[id] = previous;' in shader
    assert 'atomicExchange(heads[labelBinIndex(bin)], id)' in shader
    assert 'visited++ >= uint(pc.bin_dim_count.w)' in shader
    assert 'pc.bin_min_cap.w' not in shader
    assert shader.count('pc.padding.x != LINKED_LABEL_ABI') == 3
    assert '(flags[id] & ~pc.label_mask)' in shader
    # Actual measured 100k fixture: 1.5x grow policy on next/count/flag scratch.
    old_items = 1322093184
    next_bytes = 100000 * 4 * 3 // 2
    assert old_items - next_bytes == 1321493184
    for threads in (1, 256, 257, 65535 * 256, 65535 * 256 + 1):
        groups = (threads + 255) // 256
        x = min(groups, 65535)
        y = (groups + x - 1) // x
        assert x <= 65535 and x * y * 256 >= threads


if __name__ == '__main__':
    check_membership_and_labels()
    check_source_contracts()
    print('PASS uncapped linked membership vs all-pairs oracle, arbitrary insertion/write '
          'order, hysteresis, publication guards and N-scaled storage; no GPU execution')
