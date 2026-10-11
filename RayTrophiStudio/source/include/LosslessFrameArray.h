#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace RayTrophiSim::SimFrameCache {

// Exact object representations: no quantization of mass, heat, or solver state.
// Sparse storage is chosen only when smaller than the original dense array.
template <typename T>
class LosslessArray {
    static_assert(std::is_trivially_copyable_v<T>);

public:
    void captureAndRelease(std::vector<T>& source) {
        count_ = source.size();
        indices_.clear();
        values_.clear();
        dense_ = false;
        if (source.empty()) {
            return;
        }
        std::memcpy(&base_, &source.front(), sizeof(T));
        std::size_t changed = 0;
        for (const T& value : source) {
            changed += std::memcmp(&value, &base_, sizeof(T)) != 0;
        }
        const bool can_index = count_ <= std::numeric_limits<uint32_t>::max();
        if (!can_index || changed * (sizeof(T) + sizeof(uint32_t)) >= count_ * sizeof(T)) {
            dense_ = true;
            values_.swap(source);
            return;
        }
        indices_.reserve(changed);
        values_.reserve(changed);
        for (std::size_t i = 0; i < source.size(); ++i) {
            if (std::memcmp(&source[i], &base_, sizeof(T)) != 0) {
                indices_.push_back(static_cast<uint32_t>(i));
                values_.emplace_back();
                std::memcpy(&values_.back(), &source[i], sizeof(T));
            }
        }
        std::vector<T>().swap(source);
    }

    void restore(std::vector<T>& destination) const {
        if (dense_) {
            destination.resize(count_);
            if (count_ > 0) {
                std::memcpy(destination.data(), values_.data(), count_ * sizeof(T));
            }
            return;
        }
        destination.resize(count_);
        for (T& value : destination) {
            std::memcpy(&value, &base_, sizeof(T));
        }
        for (std::size_t i = 0; i < indices_.size(); ++i) {
            std::memcpy(&destination[indices_[i]], &values_[i], sizeof(T));
        }
    }

    std::size_t bytes() const {
        return indices_.capacity() * sizeof(uint32_t) + values_.capacity() * sizeof(T);
    }

private:
    std::size_t count_ = 0;
    bool dense_ = false;
    T base_{};
    std::vector<uint32_t> indices_;
    std::vector<T> values_;
};

} // namespace RayTrophiSim::SimFrameCache
