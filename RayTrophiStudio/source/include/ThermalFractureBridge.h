#pragma once

#include "MaterialStateField.h"
#include "RigidBodySystem.h"

namespace RayTrophiSim {

// The impulse [N.s] this group currently resists, in the units an incoming
// impulse is measured in.
//
// ★★ IT IS DERIVED FROM THE GROUP'S MASS, and that is the whole point. What an
// artist means by "how hard is this to break" is how violent a SHOVE it takes —
// a velocity — not an impulse, because an impulse means completely different
// things to a 2 kg plank and a 2 tonne tower leg. Authoring the velocity and
// multiplying by mass here makes one authored number behave the same on both,
// the same reasoning that scales melt slump by the object's own height.
//
// `group_mass_kg` is the summed mass of the group's shards. Pass <= 0 and it
// falls back to 1 kg, which reproduces the old mass-blind numbers exactly —
// useful precisely because every threshold authored before this existed was
// tuned against 1 kg shards.
float effectiveFractureThreshold(const RigidBodyObject& body,
                                 const MaterialIntegritySummary& summary,
                                 float group_mass_kg);

} // namespace RayTrophiSim
