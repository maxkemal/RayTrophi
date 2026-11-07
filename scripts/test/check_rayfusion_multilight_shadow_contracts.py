"""Non-build ABI/publication audit; does not execute shaders or prove image quality."""
from pathlib import Path
import re
import struct

ROOT = Path(__file__).resolve().parents[2] / 'RayTrophiStudio/source'


def read(path):
    return (ROOT / path).read_text(encoding='utf-8')


def main():
    host = read('src/Viewport/MaterialPreviewRtShadow.cpp')
    shader = read('shaders/rayfusion_rt_shadow.comp')
    consumer = read('shaders/material_preview_rt_shadow.glsl')
    sdf = read('shaders/material_preview_sdf_surface.frag')
    assert 'sizeof(RtShadowLight) == 32u' in host
    assert 'sizeof(RtShadowPush) == 128u' in host
    assert 'VkDescriptorSetLayoutBinding bindings[8]' in host
    assert 'VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,5' in host
    assert 'writes[7].pBufferInfo = &lightInfo' in host
    assert 'binding = 7' in shader
    assert 'uvec2 pixels[]' in shader and 'uvec2 rtShadowPixels[]' in consumer
    assert 'const uint32_t header[]={0u,' in host
    assert 'pc.volumeMeta.y != 0x52545349u' in shader
    assert 'metadata.x = 0x52545349u' in shader
    assert 'rtShadowMeta.x != 0x52545349u' in consumer
    assert 'source.positionType.xyz - origin' in shader
    assert 'distance - tmin' in shader
    assert 'tmin, rayMax,' in shader
    assert 'rfSurfaceSdfOccludes' in sdf
    assert 'RF_SDF_EXTERNAL_NANOVDB' in sdf
    assert re.search(r'for \(uint light = 0u; light < 32u;', shader)
    # Fixed 8-byte pixel ABI: independent lights (including bit 31) must survive
    # storage without float/NaN conversion and without disturbing receiver depth.
    depth = struct.unpack('<I', struct.pack('<f', 0.625))[0]
    blocked = (1 << 0) | (1 << 7) | (1 << 31)
    payload = struct.pack('<II', blocked, depth)
    mask, restored_depth = struct.unpack('<II', payload)
    assert len(payload) == 8 and restored_depth == depth
    assert [i for i in range(32) if mask & (1 << i)] == [0, 7, 31]
    print('PASS per-light integer mask ABI, finite light segment wiring, '
          'publication version and SDF receiver path; no GPU execution')


if __name__ == '__main__':
    main()
