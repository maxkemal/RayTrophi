"""Read-only cache source contracts and a byte-exact packing oracle; no C++ build."""
import math
import random
import re
import struct
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'RayTrophiStudio/source'


def pack(values):
    if not values:
        return (0, b'', [], [], False)
    base = values[0]
    different = [(i, v) for i, v in enumerate(values) if v != base]
    if len(different) * (len(base) + 4) >= len(values) * len(base):
        return len(values), base, [], values[:], True
    return len(values), base, [i for i, _ in different], [v for _, v in different], False


def unpack(image):
    count, base, indices, values, dense = image
    if dense:
        return values[:]
    output = [base] * count
    for i, v in zip(indices, values):
        output[i] = v
    return output


def main():
    rng = random.Random(42)
    for width in (1, 4, 8, 12):
        for count in (0, 1, 17, 1024):
            for ratio in (0, .01, .25, 1):
                values = [bytes(width) for _ in range(count)]
                for i in range(count):
                    if rng.random() < ratio:
                        values[i] = rng.randbytes(width)
                assert unpack(pack(values)) == values
    # Preserve signed zero, infinity and NaN payloads without numeric equality.
    bits = [struct.pack('<I', x) for x in (0, 0x80000000, 0x7fc12345, 0x7f800000)]
    assert unpack(pack(bits * 20)) == bits * 20
    grid = (SOURCE / 'include/FluidGrid.h').read_text(encoding='utf-8')
    fields = re.findall(r'^    std::vector<[^>]+>\s+(\w+)\s*;', grid, re.M)
    cache = (SOURCE / 'include/MatterGridFrameCache.h').read_text(encoding='utf-8')
    for field in fields:
        assert f'{field}.captureAndRelease(grid.{field})' in cache, field
        assert f'{field}.restore(grid.{field})' in cache, field
        assert f'grid.{field}.capacity()' in cache, field
    scene = (SOURCE / 'include/scene_data.h').read_text(encoding='utf-8')
    for token in ('c.meta = std::move(input)', 'matter_grid_cache.captureAndRelease',
                  'matter_grid_cache.restore', 'gas_mass_cache.restore',
                  'gas_energy_cache.restore', 'matter_grid_cache.bytes()',
                  'compressGridDomainStates(std::move(states)'):
        assert token in scene, token
    memory = (SOURCE / 'include/SimFrameCacheMemory.h').read_text(encoding='utf-8')
    assert 'fluidGridAllocationBytes(state.matter_liquid_grid)' in memory
    for extension in ('vcxproj', 'vcxproj.filters'):
        path = ROOT / 'RayTrophiStudio' / ('RayTrophiStudio.' + extension)
        ET.parse(path)
        for name in ('LosslessFrameArray.h', 'MatterGridFrameCache.h'):
            assert name in path.read_text(encoding='utf-8')
    print('PASS byte-exact packing oracle and all 33 grid-vector/cache ownership contracts; no C++ execution')


if __name__ == '__main__':
    main()
