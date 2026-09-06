#pragma once
#include <cmath>
#include "Vec3.h"
#include "globals.h"
#include "PostProcess/Exposure.h"
#include "PostProcess/ColorMath.h"
enum class ToneMappingType { AGX=0, ACES=1, Uncharted=2, Filmic=3, None=4, Reinhard=5 };
class ColorProcessor {
public:
    struct ColorProcessingParams {
        rtpost::ExposureSettings exposure_settings;
        float global_exposure=1.0f, global_gamma=1.0f, saturation=1.0f;
        float color_temperature=6500.0f;
        ToneMappingType tone_mapping_type=ToneMappingType::AGX;
        bool enable_vignette=true;
        float vignette_strength=0.0f;
    } params;
    float resolved_exposure=1.0f; // immutable during a pixel pass
    ColorProcessor()=default;
    ColorProcessor(int,int) {}
    void resize(int,int) {}
    void setParams(const ColorProcessingParams& p) {params=p;resolved_exposure=p.global_exposure;}
    float linearToSRGB(float x) const {
        return x<=.0031308f?12.92f*x:1.055f*std::pow(x,1.0f/2.4f)-.055f;
    }
    Vec3 processColor(const Vec3& color,int,int) const {
        auto c=rtPcGrade(RtColor(color.x*resolved_exposure,color.y*resolved_exposure,color.z*resolved_exposure),
            static_cast<int>(params.tone_mapping_type),params.color_temperature,params.saturation,params.global_gamma);
        return Vec3(c.x,c.y,c.z); // display-linear; caller applies vignette and sRGB exactly once
    }
};
