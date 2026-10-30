#ifndef VOLUME_MIXTURE_SHARED_STATE_GLSL
#define VOLUME_MIXTURE_SHARED_STATE_GLSL

// Invocation-local two-field specialization. No new GPU buffers or ABI fields.
bool mfSharedWalk;
bool mfSharedDensityReady;
vec2 mfSharedDensity;
vec2 mfSharedExtinction;

float mfSharedCameraSpan(float t, float remaining, float baseStep);
float mfSharedTemperature(uint index, vec3 position);

#endif
