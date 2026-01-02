#pragma once

/**
 * @file TerrainContext.h
 * @brief Terrain-specific context for node evaluation
 * 
 * This provides the domain context for terrain nodes,
 * allowing the new node system to work with TerrainObject.
 */

#include "NodeSystem/NodeCore.h"
#include "NodeSystem/EvaluationContext.h"

// Forward declaration
class TerrainObject;

namespace TerrainNodes {

    // ============================================================================
    // TERRAIN CONTEXT
    // ============================================================================
    
    /**
     * @brief Context for terrain node evaluation
     * 
     * Stores the terrain object and related parameters needed
     * for terrain node operations.
     */
    struct TerrainContext {
        TerrainObject* terrain = nullptr;
        
        // Resolution info (cached from terrain)
        int resolution = 0;
        float worldSize = 1000.0f;
        
        // Optional progress callback
        std::function<void(float)> progressCallback;
        
        TerrainContext() = default;
        explicit TerrainContext(TerrainObject* t) : terrain(t) {}
        
        bool isValid() const { return terrain != nullptr; }
    };

    // ============================================================================
    // TERRAIN PIN SEMANTICS
    // ============================================================================
    
    /**
     * @brief Maps NodeSystem::ImageSemantic to terrain-specific meanings
     */
    namespace TerrainSemantics {
        constexpr NodeSystem::ImageSemantic Heightmap = NodeSystem::ImageSemantic::Height;
        constexpr NodeSystem::ImageSemantic Mask = NodeSystem::ImageSemantic::Mask;
        constexpr NodeSystem::ImageSemantic Splat = NodeSystem::ImageSemantic::Generic;
    }

    // ============================================================================
    // HELPER: Get terrain from evaluation context
    // ============================================================================
    
    inline TerrainObject* getTerrainFromContext(NodeSystem::EvaluationContext& ctx) {
        if (ctx.hasDomainContext<TerrainContext>()) {
            auto* tctx = ctx.getDomainContext<TerrainContext>();
            return tctx ? tctx->terrain : nullptr;
        }
        return nullptr;
    }

    // ============================================================================
    // TERRAIN NODE CATEGORIES
    // ============================================================================
    
    namespace Categories {
        constexpr const char* Input = "Input";
        constexpr const char* Erosion = "Erosion";
        constexpr const char* Mask = "Mask";
        constexpr const char* Math = "Math";
        constexpr const char* Output = "Output";
        constexpr const char* Filter = "Filter";
    }

} // namespace TerrainNodes
