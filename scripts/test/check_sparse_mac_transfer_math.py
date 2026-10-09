"""Independent CPU oracle for MAC ownership and APIC/FLIP page contracts.

Does not execute GLSL or establish runtime/GPU parity. Tests clipped/exact tiles,
slot permutation, disjoint support, page retirement and nonzero boundary fields.
"""

import itertools
import math
import random


def shape(cells, axis):
    dims = list(cells)
    dims[axis] += 1
    return dims


def owner(face, axis, cells):
    cell = list(face)
    cell[axis] = min(cell[axis], cells[axis] - 1)
    return tuple(value // 8 for value in cell)


def address(face, axis, cells, slots):
    tile = owner(face, axis, cells)
    if tile not in slots:
        return None
    local = [value - block * 8 for value, block in zip(face, tile)]
    dims = [8, 8, 8]
    dims[axis] = 9
    return slots[tile] * 576 + local[0] + dims[0] * (local[1] + dims[1] * local[2])


def faces(cells, axis):
    return itertools.product(*(range(value) for value in shape(cells, axis)))


def stencil(position, axis, cells):
    grid = [value - (0 if index == axis else 0.5) for index, value in enumerate(position)]
    base = [math.floor(value - 0.5) for value in grid]
    weights = []
    for value, first in zip(grid, base):
        delta = value - first - 1
        weights.append((0.5 * (0.5 - delta) ** 2, 0.75 - delta ** 2,
                        0.5 * (0.5 + delta) ** 2))
    for offset in itertools.product(range(3), repeat=3):
        face = tuple(first + step for first, step in zip(base, offset))
        if any(value < 0 or value >= limit for value, limit in zip(face, shape(cells, axis))):
            continue
        weight = math.prod(weights[index][step] for index, step in enumerate(offset))
        displacement = tuple(value - center for value, center in zip(face, grid))
        yield face, weight, displacement


def ownership():
    for cells in ((1, 1, 1), (8, 8, 8), (9, 17, 7), (16, 9, 24)):
        tiles = list(itertools.product(*(range((value + 7) // 8) for value in cells)))
        random.Random(23).shuffle(tiles)
        slots = {tile: slot for slot, tile in enumerate(tiles)}
        for axis in range(3):
            seen = {address(face, axis, cells, slots): face for face in faces(cells, axis)}
            assert len(seen) == math.prod(shape(cells, axis)), (cells, axis)
            page_dims = [8, 8, 8]
            page_dims[axis] = 9
            emitted = set()
            for tile, slot in slots.items():
                for local in itertools.product(*(range(value) for value in page_dims)):
                    face = tuple(block * 8 + value for block, value in zip(tile, local))
                    if any(value >= limit for value, limit in zip(face, shape(cells, axis))):
                        continue
                    if owner(face, axis, cells) != tile:
                        continue
                    offset = slot * 576 + local[0] + page_dims[0] * (
                        local[1] + page_dims[1] * local[2])
                    assert seen[offset] == face
                    assert face not in emitted, (cells, axis, face)
                    emitted.add(face)
            assert len(emitted) == len(seen)


def transfer():
    cells = (32, 17, 9)
    positions = ((7.9, 8.1, 4.2), (8.2, 7.8, 4.1), (30.9, 16.8, 8.8),
                 (0.1, 0.2, 0.3))
    velocities = ((2, -3, 1), (-1, 4, 2), (5, 1, -2), (1, 2, 3))
    masses = (0.013, 0.021, 0.008, 0.019)
    tiles = {owner(face, axis, cells) for position in positions for axis in range(3)
             for face, _, _ in stencil(position, axis, cells)}
    slots = {tile: slot for slot, tile in enumerate(sorted(tiles, reverse=True))}
    assert len(slots) < math.prod((value + 7) // 8 for value in cells)
    for axis in range(3):
        dense_sum, dense_weight, compact_sum, compact_weight = {}, {}, {}, {}
        for position, velocity, mass in zip(positions, velocities, masses):
            for face, weight, dx in stencil(position, axis, cells):
                index = address(face, axis, cells, slots)
                assert index is not None
                affine = sum(a * b for a, b in zip((0.3, -0.2, 0.1), dx))
                for totals, weights, key in ((dense_sum, dense_weight, face),
                                             (compact_sum, compact_weight, index)):
                    totals[key] = totals.get(key, 0) + mass * weight * (velocity[axis] + affine)
                    weights[key] = weights.get(key, 0) + mass * weight
        pre_dense = {face: value / dense_weight[face] if dense_weight[face] > 1e-8 else 0
                     for face, value in dense_sum.items()}
        pre_pool = {index: value / compact_weight[index] if compact_weight[index] > 1e-8 else 0
                    for index, value in compact_sum.items()}
        # Projection/contact/moving boundaries may write faces with zero P2G
        # weight. Capture the entire allocated page, not only occupied faces.
        post_dense = {face: pre_dense.get(face, 0) + 0.017 * sum(face) - 0.4
                      for face in faces(cells, axis)}
        post_pool = {address(face, axis, cells, slots): value
                     for face, value in post_dense.items()
                     if owner(face, axis, cells) in slots}
        for position, velocity in zip(positions, velocities):
            dense_pic = sparse_pic = dense_pre = sparse_pre = 0.0
            dense_affine, sparse_affine = [0.0] * 3, [0.0] * 3
            for face, weight, dx in stencil(position, axis, cells):
                index = address(face, axis, cells, slots)
                assert abs(dense_weight.get(face, 0) - compact_weight.get(index, 0)) < 1e-15
                dense_pic += weight * post_dense[face]
                sparse_pic += weight * post_pool[index]
                dense_pre += weight * pre_dense.get(face, 0)
                sparse_pre += weight * pre_pool.get(index, 0)
                for component in range(3):
                    dense_affine[component] += weight * post_dense[face] * dx[component] * 4
                    sparse_affine[component] += weight * post_pool[index] * dx[component] * 4
            for blend in (0.0, 0.5, 0.95, 1.0):
                assert abs((dense_pic + blend * (velocity[axis] - dense_pre)) -
                           (sparse_pic + blend * (velocity[axis] - sparse_pre))) < 1e-12
            assert all(abs(a - b) < 1e-12 for a, b in zip(dense_affine, sparse_affine))
        # Full dense publication clears retired/absent pages, including weights.
        next_slots = {next(iter(slots)): 0}
        for face in faces(cells, axis):
            index = address(face, axis, cells, next_slots)
            if owner(face, axis, cells) not in next_slots:
                assert index is None


if __name__ == "__main__":
    ownership()
    transfer()
    print("PASS CPU MAC ownership, clipped/final faces, slot order and APIC/FLIP parity oracle")
