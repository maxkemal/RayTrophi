#pragma once

struct UIContext;

namespace audio {

// Per-frame scene audio, shared by Studio and the Player: transport, scene
// sync, the listener at the active camera, weather reactions, acoustics and
// cloud thunder. This used to live in the Audio panel (AudioUI::updateListener),
// which the Player does not compile, so the Player never played a sound.
// Studio passes its timeline time and playback state; the Player its game clock.
void tickRuntime(UIContext& ctx, double transport_seconds, bool playing);

} // namespace audio
