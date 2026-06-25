#pragma once

// Faz 0 link/runtime proof for the Jolt Physics integration.
// Self-contained: initializes a Jolt world, drops a sphere onto a static
// floor, steps it, and reports the settled position. Gated behind the
// --jolt-selftest CLI flag from main() so it never runs in normal sessions.
// The minimal layer-interface boilerplate here is intentionally a throwaway
// preview of what JoltWorld (Faz 1) will own properly.

namespace RayTrophiSim {
namespace JoltIntegration {

struct SmokeTestResult {
    bool initialized = false;   // Jolt allocator/factory/types registered
    bool stepped = false;       // PhysicsSystem::Update ran without crashing
    float start_y = 0.0f;       // sphere Y before stepping
    float final_y = 0.0f;       // sphere Y after settling
    int steps = 0;
};

// Runs the raw-Jolt smoke test. Returns the result and prints a one-line summary.
SmokeTestResult runSmokeTest();

// Faz 1 validation: same falling-body scenario but driven through the JoltWorld
// wrapper using RayTrophi types (Matrix4x4 transforms), exercising the adapter
// conversions (decompose/compose) end to end.
SmokeTestResult runWorldTest();

} // namespace JoltIntegration
} // namespace RayTrophiSim
