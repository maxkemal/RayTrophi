#ifndef RAYTROPHI_SIM_FORCE_FIELDS_GLSL
#define RAYTROPHI_SIM_FORCE_FIELDS_GLSL

struct PackedForceField {
    int id; int type; int shape; int falloff_type;
    uint affect_mask; int enabled;
    float position_x; float position_y; float position_z;
    float rotation_x; float rotation_y; float rotation_z;
    float scale_x; float scale_y; float scale_z;
    float strength;
    float direction_x; float direction_y; float direction_z;
    float falloff_radius; float inner_radius;
    float axis_x; float axis_y; float axis_z;
    float inward_force; float upward_force;
    float linear_drag; float quadratic_drag;
    float noise_frequency; float noise_lacunarity;
    float noise_persistence; float noise_amplitude; float noise_speed;
    int noise_octaves; int noise_seed; int use_noise;
    int fluid_surface_drag; float fluid_drag_coupling;
    float fluid_surface_depth; float fluid_curl_detail;
};

const float SIM_FORCE_DEG = 0.0174533;

vec3 simForceRotateX(vec3 p,float a){ float c=cos(a),s=sin(a); return vec3(p.x,c*p.y-s*p.z,s*p.y+c*p.z); }
vec3 simForceRotateY(vec3 p,float a){ float c=cos(a),s=sin(a); return vec3(c*p.x+s*p.z,p.y,-s*p.x+c*p.z); }
vec3 simForceRotateZ(vec3 p,float a){ float c=cos(a),s=sin(a); return vec3(c*p.x-s*p.y,s*p.x+c*p.y,p.z); }

vec3 simForceToLocal(PackedForceField f,vec3 p) {
    p-=vec3(f.position_x,f.position_y,f.position_z);
    p=simForceRotateX(p,-f.rotation_x*SIM_FORCE_DEG);
    p=simForceRotateY(p,-f.rotation_y*SIM_FORCE_DEG);
    p=simForceRotateZ(p,-f.rotation_z*SIM_FORCE_DEG);
    return p/max(abs(vec3(f.scale_x,f.scale_y,f.scale_z)),vec3(1e-6));
}

vec3 simForceToWorldVector(PackedForceField f,vec3 v) {
    v*=vec3(f.scale_x,f.scale_y,f.scale_z);
    v=simForceRotateZ(v,f.rotation_z*SIM_FORCE_DEG);
    v=simForceRotateY(v,f.rotation_y*SIM_FORCE_DEG);
    return simForceRotateX(v,f.rotation_x*SIM_FORCE_DEG);
}

bool simForceInside(PackedForceField f,vec3 p) {
    if(f.shape==0) return true;
    float r=max(f.falloff_radius,0.0);
    if(f.shape==1) return length(p)<=r;
    if(f.shape==2) return all(lessThanEqual(abs(p),vec3(r)));
    if(f.shape==3) return length(p.xz)<=r&&abs(p.y)<=r;
    if(f.shape==4) return r>1e-6&&p.y>=0.0&&p.y<=r&&length(p.xz)<=p.y;
    return true;
}

float simForceFalloff(PackedForceField f,float distance) {
    if(f.shape==0||f.falloff_type==0) return 1.0;
    float inner=max(f.inner_radius,0.0);
    float outer=max(f.falloff_radius,inner+1e-6);
    if(distance<=inner) return 1.0;
    if(distance>=outer) return 0.0;
    float t=clamp((distance-inner)/(outer-inner),0.0,1.0);
    if(f.falloff_type==1) return 1.0-t;
    if(f.falloff_type==2) return 1.0-t*t*(3.0-2.0*t);
    if(f.falloff_type==3) return sqrt(max(0.0,1.0-t*t));
    if(f.falloff_type==4) {
        float r=max(inner+t*(outer-inner),0.01),ref=max(inner,0.01);
        return (ref*ref)/(r*r);
    }
    if(f.falloff_type==5) return exp(-3.0*t);
    return 1.0-t;
}

float simForceHash31(vec3 p,float seed) {
    return fract(sin(dot(p,vec3(127.1,311.7,74.7))+seed*17.17)*43758.5453)*2.0-1.0;
}

float simForceFbm(vec3 p,PackedForceField f,float seedOffset,float timeSeconds) {
    float sum=0.0,amp=1.0,freq=max(f.noise_frequency,1e-4);
    int oct=clamp(f.noise_octaves,1,8);
    vec3 animation=vec3(timeSeconds*f.noise_speed,0.0,0.0);
    for(int i=0;i<8;++i) {
        if(i>=oct) break;
        sum+=simForceHash31((p+animation)*freq,float(f.noise_seed)+seedOffset)*amp;
        freq*=max(f.noise_lacunarity,1.0);
        amp*=clamp(f.noise_persistence,0.0,1.0);
    }
    return sum;
}

vec3 simForceCurl(vec3 p,PackedForceField f,float timeSeconds) {
    vec3 result=vec3(0.0);
    float freq=max(f.noise_frequency,1e-4),amp=1.0;
    float animation=timeSeconds*f.noise_speed+float(f.noise_seed)*0.61803398875;
    int oct=clamp(f.noise_octaves,1,8);
    for(int i=0;i<8;++i) {
        if(i>=oct) break;
        vec3 q=p*freq;
        float a=animation+float(i)*2.399963;
        result+=freq*vec3(cos(q.y+a+5.1)-cos(q.z+a+2.3),
                          cos(q.z+a+1.7)-cos(q.x+a+4.4),
                          cos(q.x+a+0.6)-cos(q.y+a+3.8))*amp;
        freq*=max(f.noise_lacunarity,1.0);
        amp*=clamp(f.noise_persistence,0.0,1.0);
    }
    return result;
}

vec3 simEvaluateForceField(PackedForceField f,vec3 worldPos,vec3 velocity,
                           float timeSeconds,uint systemMask) {
    if(f.enabled==0||(f.affect_mask&systemMask)==0u) return vec3(0.0);
    // APIC surface-drag Wind is handled by its dedicated free-surface pass.
    if(systemMask==128u&&f.type==0&&f.fluid_surface_drag!=0) return vec3(0.0);
    vec3 p=simForceToLocal(f,worldPos);
    if(!simForceInside(f,p)) return vec3(0.0);
    float dist=length(p);
    vec3 dir=vec3(f.direction_x,f.direction_y,f.direction_z);
    vec3 force=vec3(0.0);
    if(f.type==0) {
        force=dir*f.strength;
        if(f.use_noise!=0) force*=1.0+simForceFbm(worldPos,f,0.0,timeSeconds)*f.noise_amplitude;
    } else if(f.type==1) {
        force=dir*f.strength*-9.81;
    } else if(f.type==2||f.type==3) {
        if(dist>1e-3) {
            float magnitude=f.strength;
            if(f.falloff_type==4) magnitude=f.strength/(dist*dist+0.1);
            force=(p/dist)*magnitude*(f.type==2?-1.0:1.0);
        }
    } else if(f.type==4) {
        vec3 axis=vec3(f.axis_x,f.axis_y,f.axis_z);
        axis=length(axis)>1e-3?normalize(axis):vec3(0,1,0);
        vec3 radial=p-axis*dot(p,axis);
        float rd=length(radial);
        force=axis*f.upward_force;
        if(rd>1e-3) force+=normalize(cross(axis,radial))*f.strength-normalize(radial)*f.inward_force;
    } else if(f.type==5) {
        force=vec3(simForceFbm(worldPos,f,0.0,timeSeconds),
                   simForceFbm(worldPos,f,100.0,timeSeconds),
                   simForceFbm(worldPos,f,200.0,timeSeconds))*f.strength*f.noise_amplitude;
    } else if(f.type==6) {
        force=simForceCurl(worldPos,f,timeSeconds)*f.strength*f.noise_amplitude;
    } else if(f.type==7) {
        float speed=length(velocity);
        if(speed>1e-3) force=-velocity/speed*
            (f.linear_drag*speed+f.quadratic_drag*speed*speed)*f.strength;
    } else if(f.type==8) {
        if(dist>1e-3) force=cross(dir,p/dist)*f.strength;
    } else if(f.type==9) {
        force=dir*(f.strength*f.noise_amplitude*
                   simForceFbm(worldPos,f,0.0,timeSeconds));
    }
    vec3 result=simForceToWorldVector(f,force*simForceFalloff(f,dist));
    return any(isnan(result))||any(isinf(result))?vec3(0.0):result;
}

#endif
