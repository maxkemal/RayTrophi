"""Read-only lifetime contracts for late renderer-destructor logging.

Does not compile C++ or reproduce a native process shutdown.
"""
from pathlib import Path

root = Path(__file__).resolve().parents[2] / 'RayTrophiStudio/source'
source = (root / 'src/Core/globals.cpp').read_text(encoding='utf-8')
header = (root / 'include/globals.h').read_text(encoding='utf-8')
assert 'static SceneLogChangeState state;' in source
state = source.split('struct SceneLogChangeState {', 1)[1].split('};', 1)[0]
assert '~SceneLogChangeState()' in state
assert 'disableSceneLoggingForShutdown();' in state
for function in ('sceneLogOnChangeReset()',
                 'sceneLogOnChange(const std::string& key',
                 'sceneLogMessage(const std::string& msg'):
    body = source.split(function, 1)[1]
    assert 'scene_logging_available.load(std::memory_order_acquire)' in body
assert header.split('~UILogger()', 1)[1].index('disableSceneLoggingForShutdown();') < \
    header.split('~UILogger()', 1)[1].index('logFile.is_open()')
for level in ('INFO', 'WARN', 'ERROR'):
    macro = next(line for line in header.splitlines()
                 if line.startswith('#define SCENE_LOG_' + level + '('))
    assert 'sceneLogMessage(' in macro
print('PASS guarded logger/map/mutex destruction contracts; no native execution')
