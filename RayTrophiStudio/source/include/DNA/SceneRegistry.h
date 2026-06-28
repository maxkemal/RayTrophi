#pragma once

#include <vector>
#include <queue>
#include <stdexcept>
#include "EntityID.h"

namespace DNA {

    /**
     * @brief SceneRegistry manages the lifecycle and allocation of EntityIDs.
     * Implements an O(1) entity allocator with generation-based index recycling.
     */
    class SceneRegistry {
    private:
        // Stores the current generation for every index.
        // Index is the offset in this vector.
        std::vector<uint32_t> generations;

        // Queue of recycled indices that are free for allocation.
        std::queue<uint32_t> free_indices;

        // Tracks active entity count.
        size_t active_entity_count = 0;

    public:
        SceneRegistry() {
            // Reserve index 0 as null/invalid
            generations.push_back(0);
        }

        /**
         * @brief Allocates a new EntityID.
         * Recycles a free index if available, otherwise grows the registry.
         */
        EntityID create_entity() {
            uint32_t index = 0;
            uint32_t generation = 0;

            if (!free_indices.empty()) {
                index = free_indices.front();
                free_indices.pop();
                
                // Fetch current generation (already incremented on destroy)
                generation = generations[index];
            } else {
                index = static_cast<uint32_t>(generations.size());
                generation = 1; // Generation starts at 1
                generations.push_back(generation);
            }

            active_entity_count++;
            return EntityID(index, generation);
        }

        /**
         * @brief Destroys an entity, invalidating all outstanding EntityID handles.
         * Puts the index back into the recycling pool and increments its generation.
         */
        void destroy_entity(EntityID entity) {
            if (entity.is_null()) return;

            uint32_t idx = entity.index();
            if (idx >= generations.size()) {
                throw std::out_of_range("[SceneRegistry] Entity index out of bounds during destruction");
            }

            // Verify if the entity handle is valid/current
            if (generations[idx] != entity.generation()) {
                // Already destroyed or stale reference
                return;
            }

            // Increment generation to invalidate all existing handles pointing to this index
            generations[idx]++;
            if (generations[idx] == 0) {
                generations[idx] = 1; // Prevent overflow to 0 (null)
            }

            free_indices.push(idx);
            active_entity_count--;
        }

        /**
         * @brief Verifies if an EntityID handle is valid and active in the registry.
         */
        inline bool is_valid(EntityID entity) const noexcept {
            if (entity.is_null()) return false;
            
            uint32_t idx = entity.index();
            if (idx >= generations.size()) return false;
            
            return generations[idx] == entity.generation();
        }

        inline size_t active_entities() const noexcept {
            return active_entity_count;
        }

        inline size_t capacity() const noexcept {
            return generations.size();
        }

        /**
         * @brief Clears the registry, invalidating all entities.
         */
        void clear() noexcept {
            generations.clear();
            generations.push_back(0); // Reserve 0
            
            std::queue<uint32_t> empty;
            std::swap(free_indices, empty);
            
            active_entity_count = 0;
        }
    };

} // namespace DNA
