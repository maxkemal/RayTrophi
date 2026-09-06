#include "PostProcess/PostService.h"
#include "Api/RtApiInternal.h"
#include "imgui.h"
#include <cmath>
#include <cfloat>
namespace rtpost {
void drawPanel(UIContext& ctx) {
    static std::string error;
    auto apply=[&](const nlohmann::json& patch) {auto r=configure(ctx,patch);error=r.value("__error","");};
    auto check=[&](const rtapi::Result& r) {error=r.ok?"":r.error;};
    auto s=ctx.color_processor.params.exposure_settings;
    ImGui::TextUnformatted("Exposure & Color Management");
    ImGui::BeginDisabled(rtapi::renderJobActive() || ctx.render_settings.is_final_render_mode);
    const char* modes[]={"Manual EV","Physical Camera","Auto Histogram"};
    const char* keys[]={"manual","physical","auto_histogram"};
    int mode=s.mode;
    if(ImGui::Combo("Exposure Mode",&mode,modes,3)) apply({{"mode",keys[mode]}});
    if(ImGui::SliderFloat("Exposure (EV)",&s.ev,-12,12,"%.2f stops")) apply({{"ev",s.ev}});
    if(mode==1) ImGui::TextWrapped("Uses camera ISO, shutter and f-stop. Camera Auto Exposure does not override this mode.");
    if(mode==2) {
        if(ImGui::Checkbox("Lock Exposure",&s.locked)) apply({{"locked",s.locked}});
        if(s.locked && ImGui::SliderFloat("Locked Meter EV",&s.locked_ev,-24,24)) apply({{"locked_ev",s.locked_ev}});
        if(ImGui::SliderFloat("Minimum EV",&s.min_ev,-24,s.max_ev)) apply({{"min_ev",s.min_ev}});
        if(ImGui::SliderFloat("Maximum EV",&s.max_ev,s.min_ev,24)) apply({{"max_ev",s.max_ev}});
        if(ImGui::SliderFloat("Low Percentile",&s.low_percent,0,s.high_percent-.1f)) apply({{"low_percent",s.low_percent}});
        if(ImGui::SliderFloat("High Percentile",&s.high_percent,s.low_percent+.1f,100)) apply({{"high_percent",s.high_percent}});
        if(ImGui::SliderFloat("Middle Gray",&s.key,.01f,.5f)) apply({{"key",s.key}});
        if(ImGui::SliderFloat("Adapt to Bright",&s.speed_up,.1f,20,"%.1f /s")) apply({{"speed_up",s.speed_up}});
        if(ImGui::SliderFloat("Adapt to Dark",&s.speed_down,.1f,20,"%.1f /s")) apply({{"speed_down",s.speed_down}});
        if(ImGui::SliderFloat("Center Weight",&s.center_weight,0,1)) apply({{"center_weight",s.center_weight}});
        const auto t=exposureTelemetry();
        ImGui::Text("Target %.2f EV | Applied %.2f EV",t.target_ev,t.applied_ev);
        ImGui::TextWrapped("%s",t.source.c_str());
        float bars[HistogramBins];for(unsigned i=0;i<HistogramBins;++i)bars[i]=float(t.bins[i]);
        ImGui::PlotHistogram("HDR Luminance",bars,HistogramBins,0,"log2 luminance: -16 .. +16",0,FLT_MAX,ImVec2(0,65));
        if(ImGui::Button("Reset Adaptation")) {auto r=reset(ctx);error=r.value("__error","");}
    }
    const char* views[]={"AgX","ACES Fitted","Uncharted","Hejl Filmic","Linear (clip)","Reinhard"};
    const char* names[]={"agx","aces_fitted","uncharted","filmic","linear","reinhard"};
    int view=static_cast<int>(ctx.color_processor.params.tone_mapping_type);
    if(ImGui::Combo("View Transform",&view,views,6))check(rtapi::setPostToneMapping(names[view]));
    auto p=ctx.color_processor.params;
    if(ImGui::SliderFloat("White Balance (K)",&p.color_temperature,4000,25000,"%.0f"))check(rtapi::setPostColorTemperature(p.color_temperature));
    if(ImGui::SliderFloat("Saturation",&p.saturation,0,2))check(rtapi::setPostSaturation(p.saturation));
    if(ImGui::SliderFloat("Gamma / Look",&p.global_gamma,.5f,3))check(rtapi::setPostGamma(p.global_gamma));
    if(ImGui::Checkbox("Vignette",&p.enable_vignette))check(rtapi::setPostVignetteEnabled(p.enable_vignette));
    if(p.enable_vignette && ImGui::SliderFloat("Vignette Strength",&p.vignette_strength,0,2))check(rtapi::setPostVignetteStrength(p.vignette_strength));
    if(ImGui::TreeNode("Exposure Multiplier")) {
        if(ImGui::SliderFloat("Gain",&p.global_exposure,0,8))check(rtapi::setPostExposure(p.global_exposure));
        ImGui::TreePop();
    }
    ImGui::EndDisabled();
    if(!error.empty())ImGui::TextWrapped("%s",error.c_str());
}
}
