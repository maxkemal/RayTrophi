#pragma once

#include <cstdint>
#include <functional>

namespace DNA {

    /**
     * @brief EntityID representing a unique identifier in the ECS World Kernel.
     * Uses a 24-bit Generation and a 40-bit Index packed into a single 64-bit integer.
     * 
     * - Index (40 bits): Allows up to 1 trillion active/inactive entities in the scene.
     * - Generation (24 bits): Prevents ABA problems (reused indices pointing to deleted objects).
     */
    class alignas(8) EntityID {
    private:
        uint64_t id;

        static constexpr uint64_t INDEX_MASK = 0x000000FFFFFFFFFFULL;
        static constexpr uint64_t GEN_MASK   = 0xFFFFFF0000000000ULL;
        static constexpr int GEN_SHIFT = 40;

    public:
        // Default null entity
        EntityID() noexcept : id(0) {}

        // Construct from packed raw value
        explicit EntityID(uint64_t raw_id) noexcept : id(raw_id) {}

        // Construct from explicit index and generation
        EntityID(uint32_t index, uint32_t generation) noexcept {
            id = (static_cast<uint64_t>(generation) << GEN_SHIFT) | (index & INDEX_MASK);
        }

        inline uint32_t index() const noexcept {
            return static_cast<uint32_t>(id & INDEX_MASK);
        }

        inline uint32_t generation() const noexcept {
            return static_cast<uint32_t>((id & GEN_MASK) >> GEN_SHIFT);
        }

        inline uint64_t raw() const noexcept {
            return id;
        }

        inline bool is_null() const noexcept {
            return id == 0;
        }

        inline bool operator==(const EntityID& other) const noexcept {
            return id == other.id;
        }

        inline bool operator!=(const EntityID& other) const noexcept {
            return id != other.id;
        }

        inline bool operator<(const EntityID& other) const noexcept {
            return id < other.id;
        }
    };

} // namespace DNA

// Specialize std::hash for EntityID to allow its use in std::unordered_map
namespace std {
    template <>
    struct hash<DNA::EntityID> {
        size_t operator()(const DNA::EntityID& entity) const noexcept {
            return std::hash<uint64_t>{}(entity.raw());
        }
    };
}
