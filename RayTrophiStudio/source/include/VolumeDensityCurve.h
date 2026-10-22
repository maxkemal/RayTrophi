#pragma once
#include "json.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

// Fixed input knots at i/7. Monotonicity makes mapped maxima conservative.
struct VolumeDensityCurve {
    bool enabled = false;
    float values[8] = {0.0f, 1.0f/7, 2.0f/7, 3.0f/7, 4.0f/7, 5.0f/7, 6.0f/7, 1.0f};

    nlohmann::json toJson() const {
        return {{"enabled", enabled}, {"values", values}};
    }
    void fromJson(const nlohmann::json& j) {
        VolumeDensityCurve next;
        if (!j.is_object() || !j.value("enabled", nlohmann::json(false)).is_boolean()) {
            throw std::invalid_argument("density curve requires an object and boolean enabled");
        }
        for (auto it = j.begin(); it != j.end(); ++it) {
            if (it.key() != "enabled" && it.key() != "values") {
                throw std::invalid_argument("unknown density curve field: " + it.key());
            }
        }
        next.enabled = j.value("enabled", false);
        if (j.contains("values")) {
            const auto& v = j.at("values");
            if (!v.is_array() || v.size() != 8) {
                throw std::invalid_argument("density curve requires eight values");
            }
            for (int i = 0; i < 8; ++i) {
                if (!v[i].is_number()) {
                    throw std::invalid_argument("density curve values must be numeric");
                }
                float x = v[i].get<float>();
                if (!std::isfinite(x) || x < 0 || x > 1 ||
                    (i == 0 && x != 0) || (i > 0 && x < next.values[i-1])) {
                    throw std::invalid_argument("density curve must increase from zero within [0,1]");
                }
                next.values[i] = x;
            }
        }
        *this = next;
    }
    void pack(float (&out)[12]) const {
        std::fill(out, out + 12, 0.0f);
        out[0] = enabled ? 1.0f : 0.0f;
        std::copy(values, values + 8, out + 1);
    }
    float evaluate(float x) const {
        x = std::max(x, 0.0f);
        if (!enabled || x <= 0) return x;
        if (x >= 1) return x * values[7];
        float t = x * 7;
        int i = std::min(int(t), 6);
        return values[i] + (values[i+1] - values[i]) * (t-i);
    }
};
