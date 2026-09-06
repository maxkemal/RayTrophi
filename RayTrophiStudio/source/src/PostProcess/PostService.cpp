#include "PostProcess/PostService.h"
#include "Api/RtApiInternal.h"
#include "Camera.h"
#include <chrono>
#include <cmath>

namespace rtpost {
using json=nlohmann::json;
namespace {
const char* modes[]={"manual","physical","auto_histogram"};
json fail(const std::string& e) { return {{"__error",e},{"code","invalid_parameter"}}; }
bool decode(const json& patch, ExposureSettings& s, std::string& e) {
    if(!patch.is_object()) { e="settings must be an object"; return false; }
    for(auto it=patch.begin();it!=patch.end();++it) {
        const auto& k=it.key(); const auto& v=it.value();
        if(k=="mode") {
            if(!v.is_string()) {e="mode must be a string";return false;}
            s.mode=-1; for(int i=0;i<3;++i) if(v==modes[i]) s.mode=i;
        } else if(k=="locked") {
            if(!v.is_boolean()) {e="locked must be boolean";return false;} s.locked=v.get<bool>();
        } else {
            float* field=nullptr;
            if(k=="ev") field=&s.ev;
            else if(k=="min_ev") field=&s.min_ev;
            else if(k=="max_ev") field=&s.max_ev;
            else if(k=="low_percent") field=&s.low_percent;
            else if(k=="high_percent") field=&s.high_percent;
            else if(k=="key") field=&s.key;
            else if(k=="speed_up") field=&s.speed_up;
            else if(k=="speed_down") field=&s.speed_down;
            else if(k=="center_weight") field=&s.center_weight;
            else if(k=="locked_ev") field=&s.locked_ev;
            else {e="unknown exposure setting: "+k;return false;}
            if(!v.is_number()) {e=k+" must be a number";return false;} *field=v.get<float>();
        }
    }
    return validateExposure(s,e);
}
}
bool parseModernTone(const std::string& s,ToneMappingType& t) {
    if(s=="agx") t=ToneMappingType::AGX;
    else if(s=="aces_fitted") t=ToneMappingType::ACES;
    else if(s=="reinhard") t=ToneMappingType::Reinhard;
    else if(s=="linear") t=ToneMappingType::None;
    else return false;
    return true;
}
const char* modernToneName(ToneMappingType t) {
    switch(static_cast<int>(t)) {case 0:return "agx";case 1:return "aces_fitted";case 5:return "reinhard";case 4:return "linear";default:return nullptr;}
}
json saveExposure(const ExposureSettings& s) {
    return {{"mode",modes[s.mode]},{"ev",s.ev},{"min_ev",s.min_ev},{"max_ev",s.max_ev},
        {"low_percent",s.low_percent},{"high_percent",s.high_percent},{"key",s.key},
        {"speed_up",s.speed_up},{"speed_down",s.speed_down},{"center_weight",s.center_weight},
        {"locked",s.locked},{"locked_ev",s.locked_ev}};
}
void loadExposure(const json& j,ColorProcessor& p) {
    ExposureSettings s; std::string error;
    if(j.contains("exposure_v2")) { auto candidate=s; if(decode(j["exposure_v2"],candidate,error)) s=candidate; }
    p.params.exposure_settings=s;
    auto bounded=[](float v,float lo,float hi,float fallback){return std::isfinite(v)?std::clamp(v,lo,hi):fallback;};
    auto& c=p.params;
    c.global_exposure=bounded(c.global_exposure,0,65504,1);
    c.global_gamma=bounded(c.global_gamma,.1f,10,1);
    c.saturation=bounded(c.saturation,0,4,1);
    c.color_temperature=bounded(c.color_temperature,4000,25000,6500);
    c.vignette_strength=bounded(c.vignette_strength,0,2,0);
    const int type=static_cast<int>(c.tone_mapping_type);
    if(type<0 || type>5)c.tone_mapping_type=ToneMappingType::AGX;
    resetExposure();
}
json inspect(UIContext& ctx) {
    const auto t=exposureTelemetry(); const auto& p=ctx.color_processor.params;
    json out=saveExposure(p.exposure_settings);
    out["meter_supported"]=ctx.scene_ui_ptr && ctx.scene_ui_ptr->viewport_settings.shading_mode==2;
    out["schema_version"]=2; out["target_ev"]=t.target_ev; out["applied_ev"]=t.applied_ev;
    out["meter_valid"]=t.valid; out["metered_luminance"]=t.metered_luminance;
    out["meter_source"]=t.source; out["histogram"]=t.bins;
    out["histogram_min_log2"]=HistogramMin; out["histogram_max_log2"]=HistogramMax;
    out["display_exposure"]=g_display_post.exposure; out["camera_exposure"]=g_display_post.camera_exposure;
    out["tone_mapping_id"]=static_cast<int>(p.tone_mapping_type);
    out["frozen_for_render"]=rtapi::renderJobActive() || ctx.render_settings.is_final_render_mode;
    return out;
}
json configure(UIContext& ctx,const json& patch) {
    if(rtapi::renderJobActive() || ctx.render_settings.is_final_render_mode) return fail("scene is locked by the final render job");
    if(!patch.is_object()) return fail("settings must be an object");
    auto next=ctx.color_processor.params.exposure_settings;
    std::string error;
    if(!decode(patch,next,error)) return fail(error);
    if(next.locked && !ctx.color_processor.params.exposure_settings.locked && !patch.contains("locked_ev"))
        next.locked_ev=exposureTelemetry().applied_ev;
    ctx.color_processor.params.exposure_settings=next;
    updateExposure(next,0,false);
    syncDisplay(ctx.color_processor,ctx.scene.camera.get(),false);
    ctx.apply_tonemap=true; ctx.render_settings.persistent_tonemap=true;
    return inspect(ctx);
}
json reset(UIContext& ctx) {
    if(rtapi::renderJobActive() || ctx.render_settings.is_final_render_mode) return fail("scene is locked by the final render job");
    resetExposure(); syncDisplay(ctx.color_processor,ctx.scene.camera.get(),false);
    ctx.apply_tonemap=true; return inspect(ctx);
}
void syncDisplay(ColorProcessor& p,const Camera* camera,bool freeze) {
    const auto& cp=p.params;
    updateExposure(cp.exposure_settings,0,freeze || rtapi::renderJobActive());
    g_display_post.exposure=cp.global_exposure*exposureMultiplier();
    g_display_post.gamma=cp.global_gamma; g_display_post.saturation=cp.saturation;
    g_display_post.color_temperature=cp.color_temperature;
    g_display_post.vignette_strength=cp.vignette_strength;
    g_display_post.tone_mapping=static_cast<int>(cp.tone_mapping_type);
    g_display_post.vignette_enabled=cp.enable_vignette?1:0;
    float cameraExposure=1;
    if(camera && cp.exposure_settings.mode==1) {
        Camera physical=*camera; physical.auto_exposure=false; physical.use_physical_exposure=true;
        cameraExposure=physical.exposureFactor();
    }
    g_display_post.camera_exposure=cameraExposure;
    p.resolved_exposure=g_display_post.exposure*cameraExposure;
}
void tick(UIContext& ctx) {
    static auto previous=std::chrono::steady_clock::now();
    const auto now=std::chrono::steady_clock::now();
    const float dt=std::chrono::duration<float>(now-previous).count(); previous=now;
    const bool freeze=rtapi::renderJobActive() || ctx.render_settings.is_final_render_mode;
    const float before=exposureMultiplier();
    updateExposure(ctx.color_processor.params.exposure_settings,dt,freeze);
    syncDisplay(ctx.color_processor,ctx.scene.camera.get(),freeze);
    if(meterEnabled() || std::abs(exposureMultiplier()-before)>1e-6f) ctx.apply_tonemap=true;
}
}
