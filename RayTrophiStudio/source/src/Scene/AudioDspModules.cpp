// Selected upstream MIT sources are kept verbatim in external/DaisySP.
// Include them once here so both project generators discover the same modules.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "../../../external/DaisySP/Source/Synthesis/oscillator.cpp"

// M_E is a nonstandard math constant and is absent in the default MSVC headers.
// Supply it only while including the unmodified upstream ADSR implementation.
#ifndef M_E
#define M_E 2.71828182845904523536
#define RAYTROPHI_DAISY_LOCAL_M_E
#endif
#include "../../../external/DaisySP/Source/Control/adsr.cpp"
#ifdef RAYTROPHI_DAISY_LOCAL_M_E
#undef M_E
#undef RAYTROPHI_DAISY_LOCAL_M_E
#endif

#include "../../../external/DaisySP/Source/Synthesis/fm2.cpp"
#include "../../../external/DaisySP/Source/Effects/chorus.cpp"
#include "../../../external/DaisySP/Source/Effects/phaser.cpp"
#include "../../../external/DaisySP/Source/Effects/overdrive.cpp"
#include "../../../external/DaisySP/Source/PhysicalModeling/resonator.cpp"
#include "../../../external/DaisySP/Source/Filters/svf.cpp"
