#ifndef RT_PREVIEW_SHADOW_DATA
#define RT_PREVIEW_SHADOW_DATA
struct PreviewShadowRecord {
    mat4 viewProj[6];
    vec4 atlasRect[6];
    uvec4 meta;
    vec4 params;
};
layout(set=0,binding=7,std430) readonly buffer PreviewShadowBuffer {
    PreviewShadowRecord shadowRecords[33];
    uvec4 deepMeta; // atlas side, depth layers, enabled, reserved
    float deepData[];
};
layout(set=0,binding=8) uniform sampler2DShadow previewShadowAtlas;

float rtDeepTexel(ivec2 p,float depth){
    uint base=(uint(p.y)*deepMeta.x+uint(p.x))*(deepMeta.y+2u);
    float first=deepData[base],last=deepData[base+1u];
    if(last<=first || depth<=first)return 1.0;
    float u=clamp((depth-first)/(last-first),0.0,1.0)*float(deepMeta.y);
    uint hi=min(uint(floor(u)),deepMeta.y-1u);
    float a=hi==0u?0.0:deepData[base+1u+hi];
    float b=deepData[base+2u+hi];
    return exp(-mix(a,b,clamp(u-float(hi),0.0,1.0)));
}
float rtDeepShadow(vec4 rect,vec2 uv,float depth){
#ifdef RT_PREVIEW_HAS_RAYFUSION_GI_SPEC
    if(!RT_SPEC_RAYFUSION_GI)return 1.0;
#endif
    if(deepMeta.z==0u)return 1.0;
    vec2 p=(rect.xy+uv*rect.zw)*float(deepMeta.x)-0.5;
    ivec2 lo=ivec2(rect.xy*float(deepMeta.x)+0.5);
    ivec2 hi=lo+ivec2(rect.zw*float(deepMeta.x)+0.5)-1;
    ivec2 a=ivec2(floor(p));vec2 f=fract(p);
    return mix(mix(rtDeepTexel(clamp(a,lo,hi),depth),rtDeepTexel(clamp(a+ivec2(1,0),lo,hi),depth),f.x),
               mix(rtDeepTexel(clamp(a+ivec2(0,1),lo,hi),depth),rtDeepTexel(clamp(a+ivec2(1),lo,hi),depth),f.x),f.y);
}

float rtPreviewShadow(uint index,vec3 p,vec3 n,vec3 l,vec3 lightPosition,uint quality){
    if((sceneFlags&1u)==0u || index>32u)return 1.0;
    float screenVisibility=1.0;
    bool screenShadow=false;
#ifdef RT_PREVIEW_SCREEN_SHADOW
#ifdef RT_PREVIEW_HAS_RAYFUSION_GI_SPEC
    if(RT_SPEC_RAYFUSION_GI){
        screenShadow=rtScreenShadow(index,screenVisibility);
        if(screenShadow && deepMeta.z==0u)return screenVisibility;
    }
#else
    screenShadow=rtScreenShadow(index,screenVisibility);
    if(screenShadow && deepMeta.z==0u)return screenVisibility;
#endif
#endif
    uvec4 meta=shadowRecords[index].meta;
    if(meta.x==0u)return screenShadow?screenVisibility:1.0;
    uint face=0u,count=min(meta.z,6u),type=meta.y;
    if(type==0u){
        vec3 d=p-lightPosition,a=abs(d);
        face=a.x>=a.y&&a.x>=a.z?(d.x>=0?0u:1u):(a.y>=a.z?(d.y>=0?2u:3u):(d.z>=0?4u:5u));
    }
    vec3 biased=p+n*shadowRecords[index].params.y;
    float clipW=0.0;float ndcZ=0.0;vec2 uv=vec2(0);bool found=false;
    for(uint k=0u;k<6u;++k){
        uint f=type==1u?k:face;
        if(f>=count)break;
        vec4 clip=shadowRecords[index].viewProj[f]*vec4(biased,1);
        if(clip.w>0){
            vec3 ndc=clip.xyz/clip.w;vec2 candUv=ndc.xy*.5+.5;
            if(ndc.z>0&&ndc.z<1&&all(greaterThanEqual(candUv,vec2(0)))&&all(lessThanEqual(candUv,vec2(1)))){
                face=f;uv=candUv;ndcZ=ndc.z;clipW=clip.w;found=true;break;
            }
        }
        if(type!=1u)break;
    }
    if(!found)return screenShadow?screenVisibility:1.0;
    vec4 rect=shadowRecords[index].atlasRect[face];vec2 atlasUV=rect.xy+uv*rect.zw;
    if(screenShadow){
        float depth=type==1u?ndcZ:clipW;
        return screenVisibility*rtDeepShadow(rect,uv,depth);
    }
    vec4 params=shadowRecords[index].params;
    float bias=params.x*max(.25,1.0-max(dot(n,l),0.0));
    float refDepth=ndcZ-bias;
    float texel=params.z;
    float spread=max(params.w,1.0);
    vec2 lo=rect.xy+texel*.5,hi=rect.xy+rect.zw-texel*.5;
    float lit=0.0;
    if(quality<=1u && spread<=1.05){
        lit=textureLod(previewShadowAtlas,vec3(clamp(atlasUV,lo,hi),refDepth),0.0);
    }else if(quality<=2u){
        vec2 o=vec2(texel*spread*0.75);
        lit=0.25*(
            textureLod(previewShadowAtlas,vec3(clamp(atlasUV+vec2(-o.x,-o.y),lo,hi),refDepth),0.0)+
            textureLod(previewShadowAtlas,vec3(clamp(atlasUV+vec2( o.x,-o.y),lo,hi),refDepth),0.0)+
            textureLod(previewShadowAtlas,vec3(clamp(atlasUV+vec2(-o.x, o.y),lo,hi),refDepth),0.0)+
            textureLod(previewShadowAtlas,vec3(clamp(atlasUV+vec2( o.x, o.y),lo,hi),refDepth),0.0));
    }else{
        for(int y=-1;y<=1;++y)for(int x=-1;x<=1;++x){
            vec2 q=clamp(atlasUV+vec2(x,y)*texel*spread,lo,hi);
            lit+=textureLod(previewShadowAtlas,vec3(q,refDepth),0.0);
        }
        lit*=1.0/9.0;
    }
    // Apply transmittance to this light's radiance, never to ambient/emission.
    float depth=type==1u?ndcZ:clipW;
    return lit*rtDeepShadow(rect,uv,depth);
}
#endif
