"""UVW extension/smoothing and render-centre filtering oracle; no C++ build."""
import itertools
import math
from pathlib import Path
from check_surface_sparse_work import schedule

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT/'RayTrophiStudio/source'


def neighbours(c,dims):
    for a in range(3):
        for d in (-1,1):
            n=list(c);n[a]+=d;n=tuple(n)
            if 0<=n[a]<dims[a]:yield n


def evolve(field,valid,dims,work,sweeps):
    field=field.copy();valid=set(valid)
    for _ in range(sweeps):
        new=valid.copy()
        for c in work-valid:
            donors=[field[n] for n in neighbours(c,dims) if n in valid]
            if donors:
                field[c]=tuple(sum(d[a] for d in donors)/len(donors) for a in range(3))
                new.add(c)
        valid=new
    for _ in range(2):
        new=field.copy()
        for c in work&valid:
            donors=[field[n] for n in neighbours(c,dims) if n in valid]
            if donors:
                new[c]=tuple(.5*field[c][a]+.5*sum(d[a] for d in donors)/len(donors) for a in range(3))
        field=new
    return field,valid


def filtered(points,radius,amount,minimum):
    output=[]
    for p in points:
        donors=[(q,(1-math.dist(p,q)**2/radius**2)**3) for q in points if math.dist(p,q)<radius]
        if len(donors)<minimum:
            output.append(p);continue
        total=sum(w for _,w in donors)
        output.append(tuple(p[a]+amount*sum(w*(q[a]-p[a]) for q,w in donors)/total for a in range(3)))
    return output


def main():
    dims=(19,17,13);cells=set(itertools.product(*(range(n) for n in dims)))
    points=[(.2,.3,.4),(9.2,8.6,6.4),(17.8,15.5,11.8)]
    valid={c for c in cells if any(math.dist(tuple(x+.5 for x in c),p)<1.8 for p in points)}
    for sweeps in (3,5,8):
        work=schedule(dims,points,1.8+sweeps)
        for constant in ((0.,0.,0.),(.2,-.4,.8)):
            field={c:constant if c in valid else (0.,0.,0.) for c in cells}
            assert evolve(field,valid,dims,work,sweeps)==evolve(field,valid,dims,cells,sweeps)
        field={c:(c[0]*.02,c[1]*-.03,c[2]*.04) if c in valid else (0.,0.,0.) for c in cells}
        assert evolve(field,valid,dims,work,sweeps)==evolve(field,valid,dims,cells,sweeps)
    # Isolated centres, zero filter and canonical input are preserved.
    isolated=[(0.,0.,0.),(5.,0.,0.)]
    assert filtered(isolated,2.,.9,6)==isolated
    points=[(x*.3,y*.3,z*.3) for x,y,z in itertools.product(range(-2,3),repeat=3)]
    copy=points[:]
    assert filtered(points,1.,0.,6)==points
    smoothed=filtered(points,1.,.9,6)
    assert points==copy and math.dist(smoothed[len(points)//2],(0.,0.,0.))<1e-12
    for p,q in zip(points,smoothed):assert math.dist(p,q)<=.9+1e-12
    source=(SRC/'src/Physics/Fluid/FluidMaterialCoordinates.cpp').read_text(encoding='utf-8')
    assert 'SurfaceParticleBins bins' in source and 'lane < work_count' in source
    assert 'smoothSurfaceCoordinateTiles' in source and 's_smooth_scratch = uvw_out' not in source
    assert 'dp_a = particles.uvw[pa] - particles.position[pa]' in source
    assert 's_valid.swap(s_next_valid)' in source and 'extension_sweeps' in source
    surface=(SRC/'src/Physics/Fluid/FluidLevelSet.cpp').read_text(encoding='utf-8')
    assert 'prepare_centres = use_aniso || params.position_smoothing > 0.0f' in surface
    assert 'surface_bins.build(centres, particle_cell)' in surface
    assert 'const Vec3& pp = centres[pa]' in surface
    assert 'particles.position[pidx] =' not in surface
    print('PASS dense/sparse UVW parity, zero/translation residual, isolated centres and core contracts; no native execution')


if __name__=='__main__':main()
