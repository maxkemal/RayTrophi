// Non-build source/number checks. Interprets the shared scalar/vector color
// functions as JavaScript; never invokes C++, CUDA or a shader compiler.
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const root=path.resolve(import.meta.dirname,'..');
const src=path.join(root,'RayTrophiStudio/source');
const read=p=>fs.readFileSync(path.join(src,p),'utf8');
let code=read('include/PostProcess/ColorMath.h');
code=code.slice(code.indexOf('RT_PC float rtPcClamp'),code.indexOf('#undef RT_PC'));
code=code.replace(/RT_PC (?:float|RtColor) (\w+)\(([^)]*)\)/g,(_,name,args)=>
    `function ${name}(${args.replace(/\b(?:float|int|RtColor)\s+/g,'')})`);
code=code.replace(/\b(?:float|RtColor)\s+(?=\w+\s*=)/g,'let ')
    .replaceAll('RT_MIN','Math.min').replaceAll('RT_MAX','Math.max')
    .replaceAll('RT_POW','Math.pow').replaceAll('RT_LOG2','Math.log2').replace(/(\d)f\b/g,'$1');
const ctx={Math,RtColor:(x,y,z)=>({x,y,z})};vm.createContext(ctx);vm.runInContext(code,ctx);
const rgb=v=>({x:v,y:v,z:v});
const arr=c=>[c.x,c.y,c.z];
const finite=c=>arr(c).every(x=>Number.isFinite(x)&&x>=0&&x<=1);
for(let type=0;type<6;type++) {
    let prior=-1;
    for(let stop=-20;stop<=20;stop+=.25) {
        const c=ctx.rtPcGrade(rgb(.18*2**stop),type,6500,1,1);
        assert(finite(c),`finite range, type ${type}, EV ${stop}`);
        assert(c.y>=prior-5e-4,`monotonic gray, type ${type}, EV ${stop}`);prior=c.y;
        assert(Math.max(...arr(c))-Math.min(...arr(c))<.001,`neutral gray ${type}`);
    }
    for(const c of [{x:50,y:0,z:0},{x:0,y:50,z:0},{x:0,y:0,z:50},rgb(Infinity),rgb(NaN),rgb(-1)])
        for(const k of [4000,6500,25000]) assert(finite(ctx.rtPcGrade(c,type,k,4,.1)));
}
assert.deepEqual(arr(ctx.rtPcWhiteBalance(rgb(.18),6500)),[.18,.18,.18]);
assert(ctx.rtPcWhiteBalance(rgb(.18),10000).x>ctx.rtPcWhiteBalance(rgb(.18),10000).z,'higher Kelvin warms');
assert(ctx.rtPcAgX(rgb(2)).y<ctx.rtPcAgX(rgb(10)).y,'AgX preserves highlight differences');
assert(ctx.rtPcAgX(rgb(.18)).y>.1 && ctx.rtPcAgX(rgb(.18)).y<.3,'middle gray plausible');
assert.equal(ctx.rtPcTone(rgb(2),4).y,1,'linear clips');
assert(Math.abs(ctx.rtPcTone(rgb(2),5).y-2/3)<1e-8,'Reinhard is explicit');
// Interpret the actual C++ temporal adaptation function too.
let exposure=read('src/PostProcess/Exposure.cpp');
let adapt=exposure.slice(exposure.indexOf('float adaptExposure('),exposure.indexOf('ExposureSettings meterSettings'));
adapt=adapt.replace(/float adaptExposure\([^)]*\)/,'function adaptExposure(current,target,dt,s)')
    .replaceAll('std::isfinite','Number.isFinite').replaceAll('std::expm1','Math.expm1')
    .replaceAll('(std::min)','Math.min').replace(/const float /g,'const ').replace(/(\d)\.0f/g,'$1.0');
vm.runInContext(adapt,ctx);
const settings={speed_up:3,speed_down:1};
for(const target of [-6,6]) {
    const a=ctx.adaptExposure(0,target,1,settings);
    let b=0;for(let i=0;i<60;++i)b=ctx.adaptExposure(b,target,1/60,settings);
    assert(Math.abs(a-b)<1e-10,'time-step independent adaptation');
    assert(Math.abs(b)<6,'no temporal overshoot');
}
assert.equal(ctx.adaptExposure(1,6,NaN,settings),1);
for(const file of ['include/ColorProcessingParams.h','shaders/post_chain.glsl','src/Device/oidn_blend.cu'])
    assert(read(file).includes('rtPcGrade('),`${file} shares the color core`);
const vk=read('src/Backend/VulkanBackend.cpp');
assert(vk.includes('m_exposureMeter.record(cmd, hdrImage.view, slot)'));
assert(vk.includes('if (res == VK_SUCCESS) m_exposureMeter.consume(slot)'));
assert(vk.includes('pushConst.exposureFactor = 1.0f'),'HDR accumulation is exposure independent');
assert(read('src/Render/OptixWrapper.cpp').includes('launchOptixDisplayPost('));
assert(read('src/PostProcess/PostSurface.cpp').includes('if (!renderer)'),'no GPU double grading');
assert(!read('include/ColorProcessingParams.h').includes('use_adaptive_exposure'));
assert(read('src/Api/RtIpcPostExposure.cpp').includes('rtpost::configure(c,patch)'));
console.log('PASS: shared color math, HDR range, gray neutrality, highlight response, white balance, temporal adaptation and integration guards.');
