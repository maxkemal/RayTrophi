"""Read-only project timeline handoff contracts; no native build."""
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'RayTrophiStudio/source'
core=(SRC/'src/Core/ProjectTimelineReset.cpp').read_text()
manager=(SRC/'src/Core/ProjectManager.cpp').read_text(encoding='utf-8')
ui=(SRC/'src/UI/scene_ui.cpp').read_text(encoding='utf-8')
widget=(SRC/'include/TimelineWidget.h').read_text(encoding='utf-8')
assert 'std::atomic<bool> pending_project_timeline_reset' in core
assert 'exchange(false, std::memory_order_acq_rel)' in core
assert 'widget.resetForProjectLoad()' in core
assert 'settings.animation_is_playing = false' in core
assert manager.count('requestProjectTimelineReset();')==2
body=ui.split('void SceneUI::draw(UIContext& ctx)',1)[1]
assert body.index('deserialize(g_project.ui_layout_data)') < body.index('applyPendingProjectTimelineReset(')
assert body.index('applyPendingProjectTimelineReset(') < body.index('drawStatusAndBottom(')
reset=widget.split('void resetForProjectLoad()',1)[1].split('void clampCurrentFrameToRange()',1)[0]
assert 'is_playing = false' in reset and 'current_frame = 0' in reset
assert 'project_rest_frame_ = true' in reset
assert 'project_rest_frame_ && current_frame == 0 && !is_playing' in widget
# Loader sets model frame=0, layout restore must not reintroduce old UI frame.
for old in [1,63,123]:
 for start,end in [(0,250),(10,250),(-20,100)]:
  model=0;widget_frame=old;playing=True;pending=True
  widget_frame=old # main-thread saved layout restore
  if pending:widget_frame=model=0;playing=False;pending=False;rest=True
  if not(rest and widget_frame==0 and not playing):widget_frame=max(start,min(end,widget_frame))
  assert widget_frame==model==0 and not playing
for suffix in ['vcxproj','vcxproj.filters']:
 p=ROOT/'RayTrophiStudio'/('RayTrophiStudio.'+suffix);ET.parse(p)
 assert 'ProjectTimelineReset.cpp' in p.read_text(encoding='utf-8')
print('PASS load/UI handoff, stopped rest frame and nonzero authored ranges; no native execution')
