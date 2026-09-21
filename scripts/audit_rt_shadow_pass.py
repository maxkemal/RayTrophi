"""Non-build source contracts for same-frame RT shadows. --self-test tests rejection."""
import re
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent / "RayTrophiStudio/source"
FILES = {
    "core": "src/Viewport/MaterialPreviewRtShadow.cpp",
    "viewport": "src/Backend/VulkanViewportBackend.cpp",
    "legacy": "src/Backend/VulkanBackend.cpp",
    "compute": "shaders/rayfusion_rt_shadow.comp",
    "consumer": "shaders/material_preview_rt_shadow.glsl",
    "shadow": "shaders/material_preview_shadow_data.glsl",
    "frag": "shaders/material_preview_frag.frag",
    "depth": "shaders/material_preview_shadow_frag.frag",
    "uv": "shaders/material_preview_uv.glsl",
    "volume": "shaders/material_preview_volume.frag",
    "sdf": "shaders/material_preview_sdf_surface.frag",
    "api": "src/Api/RtApiViewport.cpp",
    "ipc": "src/Api/RtIpc.cpp",
    "python": "src/Api/RtPython.cpp",
    "ui": "include/UI/rayfusion_status_panel.hpp",
}
def code(text):
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)

def audit(sources):
    s={k:code(v) for k,v in sources.items()}
    failures=[]
    def need(ok, message):
        if not ok: failures.append(message)
    v=s["viewport"]; c=s["core"]; sh=s["compute"]
    need("depthOnlyPass && rtShadowFrame) resumeRtShadowShading" in v,
         "split must execute after depth and before color in the shared mesh loop")
    need("m_rasterDepthPrepassAllowed || rtShadowFrame" in v,
         "RT requires automatic depth prepass")
    need("recordRtShadowPass(" not in v, "post-pass producer must be removed")
    need(v.index("prepareRtShadowFrame(") < v.index("recordMaterialPreviewShadowPass(cmd)"),
         "consumer descriptors must be prepared before recording their first bind")
    resume=c.split("void VulkanBackendAdapter::resumeRtShadowShading",1)[-1].split("void VulkanBackendAdapter::updateRtShadowMaterialPrograms",1)[0]
    need(resume.find("vkCmdEndRenderPass") < resume.find("recordRtShadowPass") < resume.find("vkCmdBeginRenderPass"),
         "compute must run outside the render pass, before resumed shading")
    need("hdrRenderPassLoad" in resume and "load.clearValueCount=0" in resume,
         "resume must LOAD current color/depth")
    for key in ("viewport","legacy"):
        need(all(x in s[key] for x in ("mpDslBindings[24]", "mpDslci.bindingCount = 24", "mpBindingFlags[24]", "mpBindingFlagsCI.bindingCount = 24", "mpPoolSizes[0].descriptorCount = 11", "binding <= 23")),
             key+": descriptor layouts/pools must include storage binding 23")
    need("VK_ATTACHMENT_LOAD_OP_LOAD, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL" in v,
         "HDR continuation must load the stored depth")
    need("bindConsumer();clearHeader();" in c and "const uint32_t header[4]={}" in c,
         "disable/failure must invalidate old mask metadata")
    need("s.recorded=true" in c and "s.recorded && s.prepared" in c,
         "ready must reflect dispatch recording")
    need("tlas == VK_NULL_HANDLE" in c and "as.instances_skipped || as.meshes_skipped" in c,
         "missing or partial TLAS must use fallback")
    need("getRayFusionHitInstances(hits)" in c and "splitGpuMaterial" in c,
         "alpha must read canonical flat hit tables and canonical material ABI")
    need("id>=s.textureCount" in c and "m_uploadedImages.find(id)" in c,
         "unavailable opacity texture must cause fallback")
    need("m_rtShadowHasMaterialPrograms" in c and "updateRtShadowMaterialPrograms(words)" in s["legacy"],
         "graph opacity cannot silently use scalar prepass")
    need("drainInteractiveViewportInFlight()" in c, "shared compute descriptors require prior completion")
    need("gl_RayFlagsNoOpaqueEXT" in sh and "gl_RayFlagsOpaqueEXT" not in sh,
         "force candidate evaluation even for opaque BLAS")
    need("gl_RayFlagsTerminateOnFirstHitEXT" in sh and "rayQueryConfirmIntersectionEXT" in sh,
         "alpha candidates must be explicitly accepted")
    need("texelFetch(rfDepth,pix,0)" in sh and "if(depth>=1.0)return" in sh,
         "exact depth fetch and sky early exit")
    need("world.xyz/world.w" in sh, "perspective reconstruction requires division by w")
    need("uv.y=1.0-uv.y" in sh, "ray alpha UV must match raster V flip")
    need("textureLod(" in sh and "mat.opacity_tex==mat.albedo_tex" in sh,
         "compute must use explicit LOD and matching alpha channel policy")
    need(all('include "material_preview_uv.glsl"' in s[k] for k in ("compute","depth","frag")),
         "UV transform must have one shared implementation")
    need("(1u << 30u)" in s["depth"] and "opacity < 0.999" in s["depth"],
         "partial alpha/transmission must not occlude camera prepass")
    need("light != rtShadowMeta.w" in s["consumer"] and "sampleValue.y-gl_FragCoord.z" in s["consumer"],
         "mask must match both light and receiver depth")
    need("vMaterialID & 0x80000000u" in s["consumer"], "impostors require cascade fallback")
    need("#ifdef RT_PREVIEW_SCREEN_SHADOW" in s["shadow"] and
         "screenVisibility*rtDeepShadow" in s["shadow"], "preserve volume transmittance")
    need(all("material_preview_rt_shadow.glsl" not in s[k] for k in ("volume","sdf")),
         "volumes/SDF must not read mesh screen mask")
    for key,token in (("api","Result setRtShadow"),("ipc","viewport.set_rt_shadow"),
                      ("ipc","viewport.rt_shadow"),("python","rtapi::setRtShadow"),
                      ("ui","rtapi::setRtShadow"),("ui","rtapi::rtShadow")):
        need(token in s[key],key+": shared API operation missing")
    need("sizeof(RtShadowPush) == 112u" in c, "push ABI must be pinned")
    return failures

if __name__ == "__main__":
    sources={key:(ROOT/path).read_text(encoding="utf-8") for key,path in FILES.items()}
    failures=audit(sources)
    if failures:
        print("FAIL\n"+"\n".join(failures));sys.exit(1)
    if "--self-test" in sys.argv:
        for key,needle in (("viewport","depthOnlyPass && rtShadowFrame) resumeRtShadowShading"),
            ("compute","gl_RayFlagsNoOpaqueEXT"),("compute","uv.y=1.0-uv.y"),
            ("consumer","sampleValue.y-gl_FragCoord.z"),("core","const uint32_t header[4]={}")):
            mutated=dict(sources);assert needle in mutated[key]
            mutated[key]=mutated[key].replace(needle,"BROKEN")
            assert audit(mutated),"mutation escaped: "+needle
        print("PASS: five broken-source mutations rejected")
    print("PASS: same-frame order, fallback, alpha/UV, descriptor ABI, shared API/UI contracts")
