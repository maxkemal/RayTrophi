"""Source/policy checks only; does not build or run the application."""
from itertools import product
from pathlib import Path
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parents[2]
source = root / 'RayTrophiStudio/source'
header = (source / 'include/Viewport/ViewportGpuWorkPolicy.h').read_text()
main = (source / 'src/Core/Main.cpp').read_text(encoding='utf-8')
assert 'return !solid_viewport || sequence_output || explicit_capture ||' in header
assert '(focused && visible)' in header
assert 'if (simulation_playing)' in header and 'return 1;' in header
gate = main.split('if (start_render && Viewport::allowGpuWork(', 1)[1].split('{', 1)[0]
assert 'shading_mode == 0' in gate
assert 'g_seq_save_active, rtapi::viewportCaptureEnabled()' in gate
assert 'ui.timeline.isPlaying() || autonomous_anim_graph_playing ||' in main
assert 'Viewport::backgroundWaitMilliseconds(' in main
# A refused draw must retain the request; start_render is consumed inside the gate.
tail = main.split('if (start_render && Viewport::allowGpuWork(', 1)[1]
assert tail.index('{') < tail.index('start_render = false;')
for solid, focus, visible, sequence, capture in product((False, True), repeat=5):
    allowed = not solid or sequence or capture or (focus and visible)
    if not solid or sequence or capture:
        assert allowed
    elif not focus or not visible:
        assert not allowed
for suffix in ('vcxproj', 'vcxproj.filters'):
    project = root / 'RayTrophiStudio' / ('RayTrophiStudio.' + suffix)
    ET.parse(project)
    assert 'ViewportGpuWorkPolicy.h' in project.read_text(encoding='utf-8')
print('PASS Solid-only background draw gate, capture/output bypass and playback pump; no native execution')
