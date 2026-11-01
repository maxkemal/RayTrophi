"""Read the volume shader's extracted implementation for non-build source audits.

Expands only the four moved implementation files, leaving established includes
and all preprocessor conditions intact. This is not a GLSL preprocessor.
"""
from pathlib import Path
import re

EXTRACTED = {
    'volume_hit_sampling.glsl', 'volume_hit_random_walk.glsl',
    'volume_surface_shading.glsl', 'volume_fog_shading.glsl',
}


def read_volume_shader(path):
    path = Path(path)
    text = path.read_text(encoding='utf-8')
    if path.name != 'volume_closesthit.rchit':
        return text
    def expand(match):
        name = match.group(1)
        return (path.parent/name).read_text(encoding='utf-8') if name in EXTRACTED else match[0]
    return re.sub(r'#include "([^"\n]+)"', expand, text)
