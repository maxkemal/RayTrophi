// CUDA texture residency for Texture: Player (no-CUDA) link alternative.
//
// PLAYER BUILD ONLY. Studio links TextureCuda.cpp instead; the two files define
// the same four members and must never be linked into the same target. Studio
// lists this file as None. Without CUDA there is no OptiX texture object, so an
// upload reports failure and the rest are no-ops; Texture.h guards these calls
// with g_hasOptix, which is never set in the Player.
// Plan: docs/dev/RAYTROPHI_PLAYER_PLAN.md

#include "Texture.h"

bool Texture::cudaUploadFull() {
    return false;
}

void Texture::cudaUpdateFull() {
}

bool Texture::cudaUpdateRegion(int /*x*/, int /*y*/, int /*w*/, int /*h*/) {
    return false;
}

void Texture::cudaRelease(bool /*synchronize*/) {
    cuda_array = nullptr;
    tex_obj = 0;
}
