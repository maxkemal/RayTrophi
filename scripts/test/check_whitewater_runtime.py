"""Whitewater source contracts and exact bin-window oracle; no C++/GPU build."""
from pathlib import Path
import math
import random
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT/'RayTrophiStudio/source'


def neighbours(points, target, radius):
    return {i for i,p in enumerate(points) if sum((a-b)**2 for a,b in zip(p,target)) < radius**2}


def window(points, consumers, radius, size):
    positions = [p for p in points+consumers if all(0 <= v < size for v in p)]
    if not positions:
        return (0,0,0),(0,0,0)
    halo = math.ceil(radius)+1
    lo = tuple(max(0,min(math.floor(p[a]) for p in positions)-halo) for a in range(3))
    hi = tuple(min(size-1,max(math.floor(p[a]) for p in positions)+halo) for a in range(3))
    return lo,tuple(b-a+1 for a,b in zip(lo,hi))


def gather(points, target, radius, base, shape):
    bins = {}
    for i,p in enumerate(points):
        cell = tuple(math.floor(v)-b for v,b in zip(p,base))
        if all(0 <= a < n for a,n in zip(cell,shape)):
            bins.setdefault(cell,[]).append(i)
    cell = tuple(math.floor(v)-b for v,b in zip(target,base))
    reach = math.ceil(radius)
    found = set()
    for x in range(max(0,cell[0]-reach),min(shape[0]-1,cell[0]+reach)+1):
        for y in range(max(0,cell[1]-reach),min(shape[1]-1,cell[1]+reach)+1):
            for z in range(max(0,cell[2]-reach),min(shape[2]-1,cell[2]+reach)+1):
                for i in bins.get((x,y,z),[]):
                    if sum((a-b)**2 for a,b in zip(points[i],target)) < radius**2:
                        found.add(i)
    return found


def main():
    rng = random.Random(81)
    for radius in (.1,1,2.5,5):
        for location in (1,16,31):
            points = [tuple(min(31.999,max(0,location+rng.uniform(-1,1))) for _ in range(3)) for _ in range(100)]
            consumers = [tuple(min(31.999,max(0,location+rng.uniform(-2,2))) for _ in range(3)) for _ in range(15)]
            base,shape = window(points,consumers,radius,32)
            assert math.prod(shape) <= 32**3
            for target in consumers+points[:10]:
                assert gather(points,target,radius,base,shape) == neighbours(points,target,radius)
    # A falling narrow jet pays for its bounds, not an empty full cube.
    jet = [(15+rng.random(),rng.uniform(1,31),15+rng.random()) for _ in range(200)]
    _,shape = window(jet,[],2,32)
    assert math.prod(shape) < .1*32**3
    # Frozen/solid/depleted sources cannot drive whitewater generation.
    eligible = lambda model,frozen,fraction: model in ('fluid','auto-fluid') and not frozen and fraction>.02
    assert eligible('fluid',False,1)
    assert not eligible('granular',False,1) and not eligible('fluid',True,1)
    assert not eligible('fluid',False,.01)

    runtime = (SRC/'src/Physics/Fluid/WhitewaterGpuRuntime.inl').read_text(encoding='utf-8')
    assert 'have_previous' not in runtime and 'consumeGpuFoamNeighbours' not in runtime
    assert 'prepareGpuWhitewaterCurrentInputs' in runtime
    assert runtime.index('dispatchGpuFoamNeighbours(state, params') < runtime.index('expected.resize(state.particles.size())')
    assert 'overflow_fallback = overflow != 0u' in runtime
    assert 'return false;' in runtime.split('if (!batch_ok || !overflow_ok || overflow_fallback)',1)[1]
    assert 'foam_bin_allocated_cells' in runtime and 'foam_bin_window' in runtime
    assert 'FluidGpuDispatch::groups256' in runtime
    disable = (SRC/'src/Physics/Fluid/WhitewaterRuntime.cpp').read_text(encoding='utf-8')
    header = (SRC/'include/Fluid/WhitewaterRuntime.h').read_text(encoding='utf-8')
    particles = (SRC/'include/Fluid/FluidParticles.h').read_text(encoding='utf-8')
    # MSVC decorates class and struct parameter types differently. Keep the
    # forward declaration consistent across translation-unit include orders.
    assert 'class FluidParticles;' in header and 'class FluidParticles {' in particles
    assert 'struct FluidParticles;' not in header
    assert 'foam.clear()' in disable and 'compute->destroyBuffer' in disable
    assert 'substanceBirthModel(tag, *source_params' in disable
    assert '&buffers->foam_render.spheres' not in disable # graphics lifetime remains safe
    driver = (SRC/'src/Physics/Fluid/FluidDomainStep.inl').read_text(encoding='utf-8')
    enabled = driver.split('// ── Whitewater',1)[1].split('bool labels_on_gpu',1)[0]
    assert '!grid_domains_[i].fluid_foam_params.enabled' in enabled
    assert 'grid_domains_[i].fluid_foam_params.enabled' in enabled
    assert 'context.compute->synchronize()' not in enabled
    assert 'particle_residency.recover()' in enabled
    foam = (SRC/'src/Physics/Fluid/FluidFoam.cpp').read_text(encoding='utf-8')
    assert 'fluid_count > 0 &&' in foam and 'if (fluid_count == 0) return 0' in foam
    assert 'isWhitewaterFluidParcel(fluid, pi, source_params)' in foam
    registry = (SRC/'src/Device/SimulationComputeVulkan.cpp').read_text(encoding='utf-8')
    for name,bindings,size in [('sim_foam_bin_clear',2,52),('sim_foam_bin_scatter',4,52),('sim_foam_crit',6,96),('sim_foam_neigh',5,56)]:
        assert re.search(r'"'+name+r'",\s*"'+name+r'\.spv",\s*'+str(bindings)+r',\s*'+str(size),registry),name
        shader=(SRC/'shaders'/(name+'.comp')).read_text()
        assert 'offset_x' in shader and 'gl_GlobalInvocationID.y' in shader,name
    for path in ('src/Api/RtApiFluid.cpp','src/Api/RtIpc.cpp','src/Api/RtPython.cpp'):
        source=(SRC/path).read_text(encoding='utf-8')
        for field in ('inputs_current','overflow_fallback','bin_cells','bin_bytes','input_ms'):
            assert field in source,(path,field)
    for extension in ('vcxproj','vcxproj.filters'):
        path=ROOT/'RayTrophiStudio'/('RayTrophiStudio.'+extension)
        ET.parse(path)
        assert 'WhitewaterRuntime.cpp' in path.read_text(encoding='utf-8')
    print('PASS exact window-neighbour/disabled/current-input/overflow/ABI/API contracts; no GPU execution')


if __name__=='__main__':
    main()
