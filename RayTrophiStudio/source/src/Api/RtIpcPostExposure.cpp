#include "RtPostBindings.h"
#include "RtIpcMethodRegistry.h"
#include "Api/RtApiInternal.h"
#include "PostProcess/PostService.h"
#include <stdexcept>
using json=nlohmann::json;

bool dispatchPostExposureIpc(const std::string& method,const json& params,const RtIpcTemplateEnqueue& enqueue,json& out) {
    if(method=="post.get_exposure") {out=enqueue([](UIContext& c){return rtpost::inspect(c);});return true;}
    if(method=="post.configure_exposure") {
        if(!params.contains("settings") || !params["settings"].is_object()) {out={{"__error","settings must be an object"},{"code","invalid_parameter"}};return true;}
        const auto patch=params["settings"];out=enqueue([patch](UIContext& c){return rtpost::configure(c,patch);});return true;
    }
    if(method=="post.reset_exposure") {out=enqueue([](UIContext& c){return rtpost::reset(c);});return true;}
    return false;
}
