#pragma once
#include "RayFusion/Reflection.h"
namespace rtapi {
// ★★★ Bu yuzey KURAL 1'in geregi: yansima yalnizca panelden erisilebilse
//   TEST EDILEMEZ sayilirdi. Ve burada test aleti asil onemli olan sey --
//   `reflection_rays > 0` iken `reflection_shaded_hits == 0`, goruntunun
//   yansima kapaliyken uretilenle BIREBIR ayni oldugunu soyler.
RayFusion::ReflectionStatus reflectionStatus();
bool setReflection(const RayFusion::ReflectionSettings& settings, std::string& error);
}
