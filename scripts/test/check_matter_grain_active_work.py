"""Non-build layout/order checks; GPU wake/trajectory acceptance is separate."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'RayTrophiStudio/source'


def read(path):
    return (SOURCE / path).read_text(encoding='utf-8')


def main():
    shader = read('shaders/sim_matter_grain.glsl')
    active = read('shaders/sim_matter_grain_active.glsl')
    gpu = read('src/Physics/Fluid/MatterGrainGpu.cpp')
    stages = read('include/Fluid/MatterGrainStages.h')
    # The fixed-size state bank and the maximum compact list cannot alias,
    # including count < allocated capacity and a changed live count.
    for capacity in (1, 2, 255, 256, 257, 74439, 131072):
        for count in {1, max(1, capacity // 2), capacity}:
            state_end = 9 * count
            list_start, list_end = 9 * count, 10 * count
            assert state_end <= list_start <= list_end <= 10 * capacity
    assert '10 * capacity * sizeof(float)' in gpu
    assert 'scratch[9u * pc.meta.x + slot] = uintBitsToFloat(i);' in active
    assert 'i = floatBitsToUint(scratch[9u * pc.meta.x + work_index]);' in shader
    step = shader[shader.index('#elif defined(GRAIN_STEP)'):]
    # An empty queue still proves the step revision; inactive lanes return
    # before the expensive per-grain history and contact paths.
    assert step.index('diagnostics[1] = REVISION') < step.index('work_index >= diagnostics[46]')
    assert step.index('work_index >= diagnostics[46]') < step.index('uint entry = history_blocks[i]')
    assert shader.index('bool sleepOn()') < shader.index('#include "sim_matter_grain_active.glsl"')
    assert stages.index('sim_matter_grain_list_build') < stages.index('sim_matter_grain_active_classify')
    assert stages.index('sim_matter_grain_active_classify') < stages.index('sim_matter_grain_step')
    assert 'uint slot = atomicAdd(diagnostics[46], 1u);' in active
    assert '(rest & AUDIT) == 0u' in active and '!grainSleepAudit()' in active
    assert '(entry & FRESH) == 0u' in active
    assert 'grainSleepTakeAudit' not in active  # deferred requests must survive
    assert 'elapsed + pc.step_contact.x' in active
    assert 'store(i, writeBank(), p, vec3(0.0), vec3(0.0));' in active
    gate = re.search(r'const bool compact_sleep = (.*?);', gpu, re.S).group(1)
    for prerequisite in ('params.compact_sleep_work', 'sleep_on', 'wake_all == 0u',
                         '!history_reset', 'runtime.last_sleeping_grains'):
        assert prerequisite in gate
    assert gpu.index('diagnostics[45] =') < gpu.index('bool uploaded =')
    print('PASS compact work: packed-state separation, stage order, empty queue, '
          'deferred wake preservation and adaptive gate; no GPU execution')


if __name__ == '__main__':
    main()
