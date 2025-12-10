#pragma once
#include <optix.h>

// DÜZELTME: template <typename T> eklenmeli
template <typename T>
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
    __align__(OPTIX_SBT_RECORD_HEADER_SIZE) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

