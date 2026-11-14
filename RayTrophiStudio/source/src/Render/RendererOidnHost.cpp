// Renderer OIDN state: Player (no-OIDN) link alternative.
//
// PLAYER BUILD ONLY. Studio links RendererOidn.cpp, which defines the real SDK
// state and the offline denoiser methods; the two files must never be linked
// into the same target (Studio lists this one as None). The Player has no OIDN:
// the opaque state is empty and no denoiser method is linked.
// Plan: docs/dev/RAYTROPHI_PLAYER_PLAN.md (Faz 3b).

#include "Renderer.h"

struct Renderer::OidnState {};

void Renderer::OidnStateDeleter::operator()(OidnState* state) const {
    delete state;
}

Renderer::OidnStatePtr Renderer::createOidnState() {
    return OidnStatePtr(new OidnState());
}
