#pragma once

// Exact opaque pointer declarations from Embree 4. SDK definitions are needed
// only by offline implementation files, not by shared scene or hair headers.
struct RTCDeviceTy;
typedef struct RTCDeviceTy* RTCDevice;
struct RTCSceneTy;
typedef struct RTCSceneTy* RTCScene;
struct RTCBoundsFunctionArguments;
struct RTCIntersectFunctionNArguments;
struct RTCOccludedFunctionNArguments;
