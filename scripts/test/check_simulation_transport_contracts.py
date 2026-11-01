"""D2.6 source wiring/ordering audit; no build or live application access."""
from pathlib import Path
import ast
import json
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'RayTrophiStudio/source'


def read(path):
    return (SRC / path).read_text(encoding='utf-8')


def main():
    core = read('src/UI/SimulationTransport.cpp')
    playback = core[core.index('void TimelineWidget::setPlayback('):]
    assert playback.index('SimulationTransport::drain(context)') < playback.index(
        'is_playing = playing;')
    assert 'context.scene.audio->syncTransport(' in playback
    assert 'current_frame >= end_frame' in playback
    assert core.index('drain(context);') < core.index('g_sim_timeline_mode = timeline_mode;')
    assert 'context.scene.simulation_world.compute().synchronize();' in core
    assert 'render->waitForCompletion();' in core and 'viewport->waitForCompletion();' in core
    panel = read('src/UI/scene_ui_forcefield.hpp')
    assert panel.count('SimulationTransport::setTimelineMode(ui_ctx, sim_mode == 0)') == 2
    assert 'setPlayback(ctx, !is_playing);' in read('src/UI/TimelineWidget.cpp')
    api = read('src/Api/RtApiSimulationTransport.cpp')
    assert api.index('mode != "timeline" && mode != "live"') < api.index(
        'SimulationTransport::setTimelineMode(')
    assert 'ui.timeline.setPlayback(*g_ctx, playing)' in api
    assert 'simCacheStatus(cache)' in api and 'renderJobActive()' in api
    fluid_step = read('src/Api/RtApiFluid.cpp').split('Result stepFluidSimulation(float dt)', 1)[1]
    assert fluid_step.index('simulation_world.stepOnce(dt)') < fluid_step.index(
        'preserveScriptSimulationPreview()') < fluid_step.index('publishManualSimulationStep(')
    assert 'dispatchSimulationTransportIpc(method, params,' in read('src/Api/RtIpc.cpp')
    assert 'registerSimulationTransportBindings(timeline, sim)' in read('src/Api/RtPython.cpp')
    overlay = json.loads((ROOT / 'scripts/ipc_descriptor_overlay.json').read_text(encoding='utf-8'))
    for method in ('timeline.play', 'timeline.pause', 'timeline.transport',
                   'sim.get_mode', 'sim.set_mode'):
        assert method in overlay
        assert f'"{method}"' in read('src/Api/RtIpcSimulationTransport.cpp')
    ast.parse((ROOT / 'scripts/test/rt_test_simulation_transport_ipc.py').read_text())
    ns = {'ms': 'http://schemas.microsoft.com/developer/msbuild/2003'}
    for name in ('RayTrophiStudio.vcxproj', 'RayTrophiStudio.vcxproj.filters'):
        tree = ET.parse(ROOT / 'RayTrophiStudio' / name)
        compiled = [n.attrib['Include'] for n in tree.findall('.//ms:ClCompile', ns)
                    if 'Include' in n.attrib]
        for path in ('UI/SimulationTransport.cpp', 'Api/RtApiSimulationTransport.cpp',
                     'Api/RtIpcSimulationTransport.cpp', 'Api/RtPythonSimulationTransport.cpp'):
            assert compiled.count('source\\src\\' + path.replace('/', '\\')) == 1
    print('PASS: shared UI/Python/IPC transport, GPU drain ordering, validation/cache, project XML')


if __name__ == '__main__':
    main()
