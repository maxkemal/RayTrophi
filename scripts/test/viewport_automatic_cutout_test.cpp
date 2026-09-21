#include "../../RayTrophiStudio/source/include/Viewport/AutomaticCutout.h"
#include <cassert>
#include <cstdint>
struct Material {
    uint32_t flags = 0, transmission_tex = 0, opacity_tex = 7;
    float transmission = 0, opacity = 1;
};
int main() {
    Material m;
    Backend::applyAutomaticViewportCutout(m, true);
    assert(m.flags & MATERIAL_FLAG_VIEWPORT_ALPHA_CUTOUT);
    assert(!(m.flags & MATERIAL_FLAG_ALPHA_CUTOUT));
    Backend::applyAutomaticViewportCutout(m, false);
    assert(m.flags == 0);
    m.flags = MATERIAL_FLAG_ALPHA_CUTOUT;
    Backend::applyAutomaticViewportCutout(m, false);
    assert(m.flags == MATERIAL_FLAG_ALPHA_CUTOUT);
    for (int feature = 0; feature < 5; ++feature) {
        Material glass;
        if (feature == 0) glass.transmission = 0.0001f;
        if (feature == 1) glass.transmission_tex = 2;
        if (feature >= 2) glass.flags = 1u << (feature == 2 ? 17 : feature == 3 ? 19 : 24);
        Backend::applyAutomaticViewportCutout(glass, true);
        assert(!(glass.flags & MATERIAL_FLAG_VIEWPORT_ALPHA_CUTOUT));
    }
    m = Material{};
    Backend::applyAutomaticViewportCutout(m, true);
    m.transmission = 1;
    Backend::applyAutomaticViewportCutout(m, true);
    assert(!(m.flags & MATERIAL_FLAG_VIEWPORT_ALPHA_CUTOUT));
    m = Material{}; m.opacity_tex = 0;
    Backend::applyAutomaticViewportCutout(m, true);
    assert(m.flags == 0);
}
