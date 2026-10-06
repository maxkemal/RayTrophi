layout(local_size_x=256) in;
layout(set=0,binding=0) buffer Velocities { float velocity[]; };
layout(set=0,binding=1) readonly buffer StressDiag { float stress_diag[]; };
layout(set=0,binding=2) buffer StateFlags { uint flags[]; };
layout(push_constant) uniform PC {
    uint count;float dt;float contact_damping;float sleep_speed;
    float pressure_threshold;uint pad0;uint pad1;uint pad2;
#ifdef MATTER_INDEXED
    uint matter_lane;
#endif
} pc;

#ifdef MATTER_INDEXED
layout(set = 0, binding = 3) readonly buffer MatterIndices { uint matter_index[]; };
layout(set = 0, binding = 4) readonly buffer MatterCounts { uint matter_count[]; };
#endif
void main(){
    uint i = gl_GlobalInvocationID.x;
#ifdef MATTER_INDEXED
    if (i >= matter_count[pc.matter_lane]) return;
    i = matter_index[i];
#endif
    if (i >= pc.count) return;
    float pressure=max(-(stress_diag[i*3]+stress_diag[i*3+1]+stress_diag[i*3+2])/3.0,0.0);
    if(pressure<=pc.pressure_threshold)return;
    vec3 v=vec3(velocity[i*3],velocity[i*3+1],velocity[i*3+2]);
    v*=exp(-max(pc.contact_damping,0.0)*max(pc.dt,0.0));
    if(length(v)<max(pc.sleep_speed,0.0)){v=vec3(0.0);flags[i]|=8u;}else flags[i]&=~8u;
    velocity[i*3]=v.x;velocity[i*3+1]=v.y;velocity[i*3+2]=v.z;
}
