// Contact damping and sleep for MPM granular parcels, after the stress update.
// CPU twin: GranularConstitutive.h settleParticle / updateSupport.
//
// Only a SUPPORTED parcel is damped or put to sleep. Pressure alone is not
// support: a pre-stressed snow block seeded in mid air hung from its top while
// its bottom tore away, because gravity's ~0.025 m/s per substep stayed under
// sleep_speed and was zeroed again every substep (2026-11-03 live). A parcel
// resting on something does not accelerate; a free one gains g * dt. The net
// acceleration along gravity is measured against the velocity the previous
// settle left, BEFORE this step's damping (a damped fall at its terminal speed
// still shows g), and averaged over support_tau_s so the snow's tens-of-Hz rest
// jitter averages out. A fresh parcel starts unsupported. No gravity: every
// parcel counts as supported (the old rule).
layout(local_size_x=256) in;
layout(set=0,binding=0) buffer Velocities { float velocity[]; };
layout(set=0,binding=1) readonly buffer StressDiag { float stress_diag[]; };
layout(set=0,binding=2) buffer StateFlags { uint flags[]; };
layout(set=0,binding=3) buffer SupportPrev { float support_prev_vg[]; };
layout(set=0,binding=4) buffer SupportAccel { float support_accel[]; };
layout(push_constant) uniform PC {
    uint count;float dt;float contact_damping;float sleep_speed;
    float pressure_threshold;
    // 0 = damp at any speed; snow damps only slower parcels (SettleParams).
    float damping_max_speed;
    float support_tau_s;float support_tolerance;
    float gx;float gy;float gz;uint pad;
#ifdef MATTER_INDEXED
    uint matter_lane;
#endif
} pc;

#ifdef MATTER_INDEXED
layout(set = 0, binding = 5) readonly buffer MatterIndices { uint matter_index[]; };
layout(set = 0, binding = 6) readonly buffer MatterCounts { uint matter_count[]; };
#endif

const float kSupportFresh = 1.0e30;  // FluidParticles.h kGranularSupportFresh

void main(){
    uint i = gl_GlobalInvocationID.x;
#ifdef MATTER_INDEXED
    if (i >= matter_count[pc.matter_lane]) return;
    i = matter_index[i];
#endif
    if (i >= pc.count) return;
    vec3 v=vec3(velocity[i*3],velocity[i*3+1],velocity[i*3+2]);
    vec3 gravity=vec3(pc.gx,pc.gy,pc.gz);
    float g=length(gravity);
    bool supported=true;
    vec3 down=vec3(0.0);
    if(g>1e-6 && pc.dt>0.0){
        down=gravity/g;
        float vg=dot(v,down);
        float accel=support_accel[i];
        if(accel>=0.5*kSupportFresh){
            accel=g;
        }else{
            float a=(vg-support_prev_vg[i])/pc.dt;
            accel+=(a-accel)*min(pc.dt/max(pc.support_tau_s,1e-6),1.0);
        }
        support_accel[i]=accel;
        support_prev_vg[i]=vg;
        supported=accel<pc.support_tolerance*g;
    }else{
        support_accel[i]=0.0;
    }
    float pressure=max(-(stress_diag[i*3]+stress_diag[i*3+1]+stress_diag[i*3+2])/3.0,0.0);
    if(pressure<=pc.pressure_threshold || !supported){flags[i]&=~8u;return;}
    if(!(pc.damping_max_speed>0.0)||length(v)<pc.damping_max_speed)
        v*=exp(-max(pc.contact_damping,0.0)*max(pc.dt,0.0));
    if(length(v)<max(pc.sleep_speed,0.0)){v=vec3(0.0);flags[i]|=8u;}else flags[i]&=~8u;
    velocity[i*3]=v.x;velocity[i*3+1]=v.y;velocity[i*3+2]=v.z;
    // The next step measures against what this one leaves.
    if(g>1e-6) support_prev_vg[i]=dot(v,down);
}
