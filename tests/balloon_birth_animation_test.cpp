#include "../RayTrophiStudio/source/include/Paint/BalloonBirthAnimation.h"
#include <cassert>
#include <limits>

int main() {
    using Paint::balloonBirthScale;
    const auto start = balloonBirthScale(0, 0.75);
    assert(start.radial > 0 && start.radial < 0.1f);
    assert(start.vertical > 0 && start.vertical < 0.1f);
    for (const double age : {0.75, 1.0, 100.0}) {
        const auto end = balloonBirthScale(age, 0.75);
        assert(end.radial == 1.0f && end.vertical == 1.0f);
    }
    const auto disabled = balloonBirthScale(0, 0);
    assert(disabled.radial == 1.0f && disabled.vertical == 1.0f);
    bool overshoot = false;
    bool squash_stretch = false;
    for (int frame = 0; frame <= 1000; ++frame) {
        const auto scale = balloonBirthScale(frame * 0.00075, 0.75);
        assert(std::isfinite(scale.radial) && std::isfinite(scale.vertical));
        assert(scale.radial > 0.05f && scale.radial < 1.25f);
        assert(scale.vertical > 0.05f && scale.vertical < 1.25f);
        overshoot |= scale.radial > 1.02f || scale.vertical > 1.02f;
        squash_stretch |= std::abs(scale.radial - scale.vertical) > 0.01f;
    }
    assert(overshoot && squash_stretch);
    const auto before_end = balloonBirthScale(0.7499, 0.75);
    assert(std::abs(before_end.radial - 1) < 1e-4f);
    assert(std::abs(before_end.vertical - 1) < 1e-4f);
    const auto invalid = balloonBirthScale(std::numeric_limits<double>::quiet_NaN(), 1);
    assert(invalid.radial == 1 && invalid.vertical == 1);
}
