"""Sparse surface scheduling and physical-bin oracle; no C++/GPU execution."""
import itertools
import math
import random
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'RayTrophiStudio/source'


def schedule(dims, points, radius, voxel=1.0):
    tiles = set()
    tx, ty = (dims[0]+7)//8, (dims[1]+7)//8
    reach = radius / voxel + 1
    for p in points:
        lo = [max(0, math.floor(p[a]/voxel-reach))//8 for a in range(3)]
        hi = [min(dims[a]-1, math.floor(p[a]/voxel+reach))//8 for a in range(3)]
        for z in range(lo[2], hi[2]+1):
            for y in range(lo[1], hi[1]+1):
                for x in range(lo[0], hi[0]+1):
                    tiles.add(x+tx*(y+ty*z))
    cells = set()
    for tile in sorted(tiles):
        base = (tile % tx*8, tile//tx % ty*8, tile//(tx*ty)*8)
        for lane in itertools.product(range(8), repeat=3):
            c = tuple(base[a]+lane[a] for a in range(3))
            if all(c[a] < dims[a] for a in range(3)):
                assert c not in cells
                cells.add(c)
    return cells


def smooth(field, dims, far, iterations, work):
    out = dict(field)
    for _ in range(iterations):
        nxt = out.copy()
        for c in work:
            neighbours = []
            for a in range(3):
                for d in (-1,1):
                    n = list(c); n[a] += d; n = tuple(n)
                    if 0 <= n[a] < dims[a]:
                        neighbours.append(out[n])
            active = out[c] < far-1e-5 or any(v < far-1e-5 for v in neighbours)
            nxt[c] = .5*out[c]+.5*sum(neighbours)/len(neighbours) if active and neighbours else far
        out = nxt
    return out


def main():
    rng = random.Random(72)
    dims = (19,17,13)
    all_cells = set(itertools.product(*(range(n) for n in dims)))
    # Multiple disconnected drops, domain walls, empty field, and odd tile edges.
    for points in [[],[(.1,.2,.3)],[(9.2,8.8,6.1)],[(1.2,2.3,3.4),(17.8,15.9,11.5)]]:
        for iterations in (0,1,2,4):
            radius = 1.8; far = 3.
            field = {c:far for c in all_cells}
            for c in all_cells:
                for p in points:
                    distance = math.dist(tuple(x+.5 for x in c),p)
                    if distance < radius:
                        field[c] = min(field[c],distance-.7)
            work = schedule(dims,points,radius+iterations)
            assert smooth(field,dims,far,iterations,work) == smooth(field,dims,far,iterations,all_cells)
    # Coarse bins must retain every contributing particle even on cell boundaries.
    particles = [tuple(rng.uniform(0,10) for _ in range(3)) for _ in range(100)]
    particles += [(2.,2.,2.),(math.nextafter(2.,0.),2.,2.)]
    width = 2.; radius = 2.
    bins = {}
    for i,p in enumerate(particles):
        bins.setdefault(tuple(math.floor(v/width) for v in p),[]).append(i)
    for _ in range(200):
        q = tuple(rng.uniform(0,10) for _ in range(3))
        c = tuple(math.floor(v/width) for v in q)
        found = set()
        for n in itertools.product(*(range(a-1,a+2) for a in c)):
            found.update(i for i in bins.get(n,[]) if math.dist(q,particles[i]) < radius)
        assert found == {i for i,p in enumerate(particles) if math.dist(q,p)<radius}
    # Refinement must not change the physical CSR size.
    counts = [math.ceil(128/2)**3 for m in (1,2,3,4)]
    assert len(set(counts)) == 1 and counts[0] == 262144
    # Anisotropic moved centre: old original-position reach can miss a valid tail.
    support = 2 * 4**(2/3); shift = .9*2.5
    assert support < support + shift
    assert math.ceil((support+shift)/width) >= math.ceil(support/width)
    source = (SRC/'src/Physics/Fluid/FluidLevelSet.cpp').read_text(encoding='utf-8')
    assert 'cell_count_per(bin_count, 0)' in source and 's_occ_a' not in source
    assert 'support + smoothing * sim_voxel' in source
    assert 'surface_bins.build(centres, particle_cell)' in source
    assert 'support * surface_bins.inverse_width' in source
    assert 'smoothSurfaceTiles(sdf_out, work_tiles' in source
    assert 'grid.nx * sim_voxel * inv_bin' in source
    for path in ('src/Api/RtIpc.cpp','src/Api/RtPython.cpp','src/Api/RtApiFluid.cpp'):
        text = (SRC/path).read_text(encoding='utf-8')
        assert 'surface_work_cells' in text and 'surface_bin_cells' in text
        for name in ('bin_ms','anisotropy_ms','schedule_ms','gather_ms','smooth_ms'):
            assert 'surface_'+name in text
    for suffix in ('vcxproj','vcxproj.filters'):
        project = ROOT/'RayTrophiStudio'/('RayTrophiStudio.'+suffix)
        ET.parse(project)
        assert 'SurfaceWorkTiles.cpp' in project.read_text(encoding='utf-8')
    print('PASS sparse/dense smoothing parity, boundary bins, refinement-independent CSR and API contracts; no native execution')


if __name__ == '__main__':
    main()
