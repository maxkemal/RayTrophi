// Shared C++ / CUDA / GLSL scene-linear Rec.709 display math.
// AgX numerical constants: three.js / Filament AgX implementation (MIT/Apache-2.0).
// See docs/dev/POST_PROCESS_V2.md for reference and licensing details.
#ifndef RT_POST_COLOR_MATH
#define RT_POST_COLOR_MATH
#ifdef __cplusplus
#include <cmath>
#ifdef __CUDACC__
#define RT_PC __host__ __device__ inline
#else
#define RT_PC inline
#endif
struct RtColor {
    float x,y,z;
    RT_PC RtColor(float a,float b,float c):x(a),y(b),z(c) {}
};
#define RT_POW powf
#define RT_LOG2 log2f
#define RT_MIN fminf
#define RT_MAX fmaxf
#else
#define RT_PC
#define RtColor vec3
#define RT_POW pow
#define RT_LOG2 log2
#define RT_MIN min
#define RT_MAX max
#endif
RT_PC float rtPcClamp(float x) { return RT_MIN(RT_MAX(x,0.0f),1.0f); }
RT_PC RtColor rtPcClip(RtColor c) { return RtColor(rtPcClamp(c.x),rtPcClamp(c.y),rtPcClamp(c.z)); }
RT_PC RtColor rtPcMatrix(RtColor c, RtColor a, RtColor b, RtColor d) {
    return RtColor(c.x*a.x+c.y*a.y+c.z*a.z,c.x*b.x+c.y*b.y+c.z*b.z,c.x*d.x+c.y*d.y+c.z*d.z);
}
RT_PC float rtPcSigmoid(float v) {
    float x = rtPcClamp((RT_LOG2(RT_MAX(v,1e-10f))+12.47393f)/16.5f);
    float x2=x*x, x4=x2*x2;
    return 15.5f*x4*x2-40.14f*x4*x+31.96f*x4-6.868f*x2*x+.4298f*x2+.1191f*x-.00232f;
}
RT_PC RtColor rtPcAgX(RtColor c) {
    c=rtPcMatrix(c,RtColor(.6274f,.3293f,.0433f),RtColor(.0691f,.9195f,.0113f),RtColor(.0164f,.0880f,.8956f));
    c=rtPcMatrix(c,RtColor(.8566271533f,.0951212405f,.0482516061f),RtColor(.1373189729f,.7612419906f,.1014390365f),RtColor(.1118982130f,.0767994186f,.8113023684f));
    c=RtColor(rtPcSigmoid(c.x),rtPcSigmoid(c.y),rtPcSigmoid(c.z));
    c=rtPcMatrix(c,RtColor(1.1271005818f,-.1106066431f,-.0164939387f),RtColor(-.1413297635f,1.1578237022f,-.0164939387f),RtColor(-.1413297635f,-.1106066431f,1.2519364066f));
    c=RtColor(RT_POW(RT_MAX(c.x,0.0f),2.2f),RT_POW(RT_MAX(c.y,0.0f),2.2f),RT_POW(RT_MAX(c.z,0.0f),2.2f));
    return rtPcClip(rtPcMatrix(c,RtColor(1.6605f,-.5876f,-.0728f),RtColor(-.1246f,1.1329f,-.0083f),RtColor(-.0182f,-.1006f,1.1187f)));
}
RT_PC float rtPcAcesFit(float x) { return (x*(x+.0245786f)-.000090537f)/(x*(.983729f*x+.4329510f)+.238081f); }
RT_PC RtColor rtPcAces(RtColor c) {
    c=rtPcMatrix(c,RtColor(.59719f,.35458f,.04823f),RtColor(.07600f,.90834f,.01566f),RtColor(.02840f,.13383f,.83777f));
    c=RtColor(rtPcAcesFit(c.x),rtPcAcesFit(c.y),rtPcAcesFit(c.z));
    return rtPcClip(rtPcMatrix(c,RtColor(1.60475f,-.53108f,-.07367f),RtColor(-.10208f,1.10813f,-.00605f),RtColor(-.00327f,-.07276f,1.07602f)));
}
// Daylight-locus white point; 6500 K is EXACT identity. Bradford D65->target
// is an artistic warming convention (higher K warms), matching the old dial.
RT_PC RtColor rtPcWhiteLms(float kelvin) {
    float t=RT_MIN(RT_MAX(kelvin,4000.0f),25000.0f);
    float x=t<=7000.0f ? -4.6070e9f/(t*t*t)+2.9678e6f/(t*t)+.09911e3f/t+.244063f
                       : -3.0258469e9f/(t*t*t)+2.1070379e6f/(t*t)+.2226347e3f/t+.240390f;
    float y=-3.0f*x*x+2.87f*x-.275f;
    return rtPcMatrix(RtColor(x/y,1.0f,(1.0f-x-y)/y),RtColor(.8951f,.2664f,-.1614f),RtColor(-.7502f,1.7135f,.0367f),RtColor(.0389f,-.0685f,1.0296f));
}
RT_PC RtColor rtPcWhiteBalance(RtColor c,float kelvin) {
    if (kelvin==6500.0f) return c;
    RtColor w=rtPcWhiteLms(kelvin), d=rtPcWhiteLms(6500.0f);
    c=rtPcMatrix(c,RtColor(.4124564f,.3575761f,.1804375f),RtColor(.2126729f,.7151522f,.0721750f),RtColor(.0193339f,.1191920f,.9503041f));
    c=rtPcMatrix(c,RtColor(.8951f,.2664f,-.1614f),RtColor(-.7502f,1.7135f,.0367f),RtColor(.0389f,-.0685f,1.0296f));
    c=RtColor(c.x*d.x/w.x,c.y*d.y/w.y,c.z*d.z/w.z);
    c=rtPcMatrix(c,RtColor(.9869929f,-.1470543f,.1599627f),RtColor(.4323053f,.5183603f,.0492912f),RtColor(-.0085287f,.0400428f,.9684867f));
    c=rtPcMatrix(c,RtColor(3.2404542f,-1.5371385f,-.4985314f),RtColor(-.9692660f,1.8760108f,.0415560f),RtColor(.0556434f,-.2040259f,1.0572252f));
    return RtColor(RT_MAX(c.x,0.0f),RT_MAX(c.y,0.0f),RT_MAX(c.z,0.0f));
}
RT_PC float rtPcUnchartedCurve(float x) {
    return ((x*(.15f*x+.05f)+.004f)/(x*(.15f*x+.5f)+.06f))-.02f/.30f;
}
RT_PC float rtPcFilmicCurve(float c) {
    float x=RT_MAX(c-.004f,0.0f);
    return RT_POW((x*(6.2f*x+.5f))/(x*(6.2f*x+1.7f)+.06f),2.2f);
}
RT_PC RtColor rtPcTone(RtColor c,int type) {
    if(type==0) return rtPcAgX(c);
    if(type==1) return rtPcAces(c);
    if(type==2) {
        float w=rtPcUnchartedCurve(11.2f);
        return RtColor(rtPcUnchartedCurve(c.x*2.0f)/w,rtPcUnchartedCurve(c.y*2.0f)/w,rtPcUnchartedCurve(c.z*2.0f)/w);
    }
    if(type==3) return RtColor(rtPcFilmicCurve(c.x),rtPcFilmicCurve(c.y),rtPcFilmicCurve(c.z));
    if(type==5) return RtColor(c.x/(1.0f+c.x),c.y/(1.0f+c.y),c.z/(1.0f+c.z));
    return rtPcClip(c); // 4: explicit linear/clipping display
}
RT_PC float rtPcSafe(float x) { return x!=x ? 0.0f : RT_MIN(RT_MAX(x,0.0f),65504.0f); }
RT_PC RtColor rtPcGrade(RtColor c,int type,float kelvin,float saturation,float gamma) {
    c=RtColor(rtPcSafe(c.x),rtPcSafe(c.y),rtPcSafe(c.z));
    c=rtPcWhiteBalance(c,kelvin);
    c=rtPcTone(c,type);
    float lum=.2126f*c.x+.7152f*c.y+.0722f*c.z;
    float invGamma=1.0f/RT_MAX(gamma,.1f);
    return rtPcClip(RtColor(RT_POW(RT_MAX(lum+(c.x-lum)*saturation,0.0f),invGamma),
        RT_POW(RT_MAX(lum+(c.y-lum)*saturation,0.0f),invGamma),
        RT_POW(RT_MAX(lum+(c.z-lum)*saturation,0.0f),invGamma)));
}
#undef RT_PC
#undef RT_POW
#undef RT_LOG2
#undef RT_MIN
#undef RT_MAX
#ifndef __cplusplus
#undef RtColor
#endif
#endif
