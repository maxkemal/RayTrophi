#pragma once
#include "imgui.h"
#include <algorithm>
#include <cmath>
namespace ScalarFieldOverlay {
inline ImU32 color(float value) {const float v=std::isfinite(value)?std::clamp(value,0.f,1.f):0.f;return IM_COL32(70,150,240,static_cast<int>(v*v*165.f));}
inline void triangle(ImDrawList* draw,const ImVec2& a,const ImVec2& b,const ImVec2& c,float wa,float wb,float wc) {
    const auto uv=ImGui::GetFontTexUvWhitePixel();draw->PrimReserve(3,3);
    draw->PrimVtx(a,uv,color(wa));draw->PrimVtx(b,uv,color(wb));draw->PrimVtx(c,uv,color(wc));
}
}
