import sys

with open('e:/RayTrophi_projesi/raytracing_Proje_Moduler/raytrac_sdl2/source/shaders/PNanoVDB.h', 'r', encoding='utf-8') as f:
    text = f.read()

import re

# We see that glslc is rejecting our array initializers heavily.
# We completely strip out:
# PNANOVDB_STATIC_CONST pnanovdb_uint32_t ...
# and PNANOVDB_STATIC_CONST pnanovdb_grid_type_constants_t ...
# And redefine PNANOVDB_GRID_TYPE_GET directly as a switch-case-like function or simplified macro!

text = re.sub(r'PNANOVDB_STATIC_CONST pnanovdb_uint32_t ([a-zA-Z0-9_]+)\[PNANOVDB_GRID_TYPE_CAP\][^;]*;', '', text)
text = re.sub(r'PNANOVDB_STATIC_CONST pnanovdb_grid_type_constants_t ([a-zA-Z0-9_]+)\[PNANOVDB_GRID_TYPE_CAP\][^;\}]+\}\s*\)\s*;', '', text)
text = re.sub(r'PNANOVDB_STATIC_CONST pnanovdb_grid_type_constants_t ([a-zA-Z0-9_]+)\[PNANOVDB_GRID_TYPE_CAP\][^;]*;', '', text)


# We also need to fix line 1948 error: pnanovdb_tree_handle_t ... { ... };
# Re-apply our fix for handle struct initializers:
text = re.sub(r'([a-zA-Z0-9_]+_handle_t)\s+([a-zA-Z0-9_]+)\s*=\s*\{\s*([^}]+)\s*\};',
              r'\1 \2; \2.address = \3;', text)


# Replace PNANOVDB_GRID_TYPE_GET fully again, just in case:
replacement = r'''
// Replaced macro for glsl compilation
#define PNANOVDB_GRID_TYPE_GET_value_strides_bits 0u
#define PNANOVDB_GRID_TYPE_GET_table_strides_bits 1u
#define PNANOVDB_GRID_TYPE_GET_minmax_strides_bits 2u
#define PNANOVDB_GRID_TYPE_GET_minmax_aligns_bits 3u
#define PNANOVDB_GRID_TYPE_GET_stat_strides_bits 4u
#define PNANOVDB_GRID_TYPE_GET_leaf_type 5u

uint pnanovdb_get_grid_attr(uint t, uint attr) {
    if (attr == 0u) return 32u;
    if (attr == 1u) return 64u;
    if (attr == 2u) return 32u;
    if (attr == 3u) return 32u;
    if (attr == 4u) return 32u;
    if (attr == 5u) return 0u;
    return 0u; // fallback
}
#define PNANOVDB_GRID_TYPE_GET(gridType, nameIn) pnanovdb_get_grid_attr(gridType, PNANOVDB_GRID_TYPE_GET_##nameIn)
'''

text = re.sub(r'^#define PNANOVDB_GRID_TYPE_GET.*?$', replacement, text, flags=re.MULTILINE)


with open('e:/RayTrophi_projesi/raytracing_Proje_Moduler/raytrac_sdl2/source/shaders/PNanoVDB.h', 'w', encoding='utf-8') as f:
    f.write(text)

print('Success.')
