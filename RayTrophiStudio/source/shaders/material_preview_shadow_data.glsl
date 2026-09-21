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
layout(set=0,binding=8) uniform sampler2D previewShadowAtlas;

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
    screenShadow=rtScreenShadow(index,screenVisibility);
    if(screenShadow && deepMeta.z==0u)return screenVisibility;
#endif
    if(shadowRecords[index].meta.x==0u)return screenShadow?screenVisibility:1.0;
    uint face=0u,count=min(shadowRecords[index].meta.z,6u),type=shadowRecords[index].meta.y;
    if(type==0u){
        vec3 d=p-lightPosition,a=abs(d);
        face=a.x>=a.y&&a.x>=a.z?(d.x>=0?0u:1u):(a.y>=a.z?(d.y>=0?2u:3u):(d.z>=0?4u:5u));
    }
    vec3 biased=p+n*shadowRecords[index].params.y;
    vec4 clip=vec4(0);vec3 ndc=vec3(0);vec2 uv=vec2(0);bool found=false;
    for(uint k=0u;k<6u;++k){
        uint f=type==1u?k:face;
        if(f>=count)break;
        clip=shadowRecords[index].viewProj[f]*vec4(biased,1);
        if(clip.w>0){
            ndc=clip.xyz/clip.w;uv=ndc.xy*.5+.5;
            if(ndc.z>0&&ndc.z<1&&all(greaterThanEqual(uv,vec2(0)))&&all(lessThanEqual(uv,vec2(1)))){face=f;found=true;break;}
        }
        if(type!=1u)break;
    }
    if(!found)return screenShadow?screenVisibility:1.0;
    vec4 rect=shadowRecords[index].atlasRect[face];vec2 atlasUV=rect.xy+uv*rect.zw;
    if(screenShadow){
        float depth=type==1u?ndc.z:clip.w;
        return screenVisibility*rtDeepShadow(rect,uv,depth);
    }
    float bias=shadowRecords[index].params.x*max(.25,1.0-max(dot(n,l),0.0));
    float lit=0.0,texel=shadowRecords[index].params.z;int radius=quality>=3u?2:1;
    vec2 lo=rect.xy+texel*.5,hi=rect.xy+rect.zw-texel*.5;
    for(int y=-radius;y<=radius;++y)for(int x=-radius;x<=radius;++x){
        vec2 q=clamp(atlasUV+vec2(x,y)*texel*max(shadowRecords[index].params.w,1.0),lo,hi);
        lit+=ndc.z-bias<=texture(previewShadowAtlas,q).r?1.0:0.0;
    }
    // Apply transmittance to this light's radiance, never to ambient/emission.
    float depth=type==1u?ndc.z:clip.w;
    return lit/float((2*radius+1)*(2*radius+1))*rtDeepShadow(rect,uv,depth);
}
#endif
