// Standalone core regression test. The user builds/runs this; agent does not compile.
#include "../../RayTrophiStudio/source/include/VolumeTemperatureMapping.h"
#include <cassert>
#include <cmath>
#include <limits>
#include <initializer_list>

int main() {
    assert(volumeResolveTemperatureUnits(0, true) == 1);
    assert(volumeResolveTemperatureUnits(0, false) == 0);
    assert(volumeResolveTemperatureUnits(2, true) == 2);

    // Low positive Kelvin values must never jump into the normalized branch.
    for (float t : {0.1f, 1.0f, 5.0f, 20.0f, 20.01f, 100.0f, 1340.0f}) {
        assert(!volumeMapTemperature(t, 30.0f, 1340.0f, 20000.0f, 1, true).valid);
    }
    auto below = volumeMapTemperature(20.0f, 1.0f, 0.0f, 20000.0f, 1, true);
    auto above = volumeMapTemperature(20.01f, 1.0f, 0.0f, 20000.0f, 1, true);
    assert(below.valid && above.valid);
    assert(std::abs(above.kelvin - below.kelvin - 0.01f) < 1e-4f);

    // Explicit fields are authoritative, independent of arbitrary density values.
    for (int units : {1, 2}) {
        assert(!volumeMapTemperature(0.0f, 30.0f, 0.0f, 20000.0f, units, true).valid);
        assert(!volumeMapTemperature(5000.0f, 30.0f, 0.0f, 20000.0f, units, false).valid);
        assert(!volumeMapTemperature(std::numeric_limits<float>::quiet_NaN(),
            30.0f, 0.0f, 20000.0f, units, true).valid);
        assert(!volumeMapTemperature(std::numeric_limits<float>::infinity(),
            30.0f, 0.0f, 20000.0f, units, true).valid);
    }
    auto normalized = volumeMapTemperature(0.5f, 0.0f, 1000.0f, 5000.0f, 2, true);
    assert(normalized.valid && normalized.kelvin == 3000.0f);
    assert(normalized.normalized == 0.5f);
    auto legacy = volumeMapTemperature(0.0f, 0.5f, 1000.0f, 5000.0f, 0, false);
    assert(legacy.valid && legacy.kelvin == 3000.0f);

    float previous = 0.0f;
    for (int i = 1; i <= 22000; ++i) {
        auto value = volumeMapTemperature(float(i), 0.1f, 0.0f, 20000.0f, 1, true);
        auto other = volumeMapTemperature(float(i), 100.0f, 0.0f, 20000.0f, 1, true);
        assert(value.valid && value.kelvin >= previous && value.kelvin <= 20000.0f);
        assert(value.kelvin == other.kelvin);
        previous = value.kelvin;
    }
    assert(!volumeMapTemperature(1.0f, 0.0f, 5.0f, 1.0f, 1, true).valid);
    assert(!volumeMapTemperature(1.0f, 0.0f, 0.0f, 10.0f, 9, true).valid);
}
