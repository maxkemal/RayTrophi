#pragma once

struct UIContext;

namespace ParticleSystemAuthoringUI {

void drawCreationBar(UIContext& ctx,
                     int& selected_emitter_index,
                     int& selected_collider_index,
                     int& selected_domain_index);

} // namespace ParticleSystemAuthoringUI
