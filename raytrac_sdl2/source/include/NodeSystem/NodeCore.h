/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          NodeCore.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once

/**
 * @file NodeCore.h
 * @brief Core type definitions for the node graph system
 * 
 * This file contains domain-agnostic types that can be used by any
 * node-based system (Terrain, Material, Particle, etc.)
 * 
 * Inspired by Gaea's architecture with:
 * - Strongly typed pin values
 * - Pull-based evaluation
 * - Extensible data types
 */

#include <variant>
#include <memory>
#include <vector>
#include <array>
#include <string>
#include <functional>
#include <any>
#include <unordered_map>
#include <atomic>
#include "imgui.h"

namespace NodeSystem {

    // ============================================================================
    // DATA TYPES
    // ============================================================================
    
    /**
     * @brief Enumeration of core data types that can flow through pins
     * 
     * These are the fundamental types. Domain-specific extensions can use
     * Custom + a subtype identifier, or domains can interpret Image2D
     * differently (heightmap vs texture vs mask).
     */
    enum class DataType : uint8_t {
        None = 0,       ///< No data / unconnected
        Float,          ///< Single floating point value
        Int,            ///< Single integer value
        Bool,           ///< Boolean flag
        Vector2,        ///< 2D vector (UV, position)
        Vector3,        ///< 3D vector (position, direction, color RGB)
        Vector4,        ///< 4D vector (quaternion, color RGBA)
        Color,          ///< RGBA color (0-1 range)
        Image2D,        ///< 2D data array (heightmap, mask, texture channel)
        String,         ///< Text data
        Custom = 255    ///< Domain-specific extension point
    };

    /**
     * @brief Semantic hint for Image2D data to help with visualization
     */
    enum class ImageSemantic : uint8_t {
        Generic = 0,    ///< Unknown purpose
        Height,         ///< Heightmap data (terrain)
        Mask,           ///< Mask/weight data
        Normal,         ///< Normal map
        Albedo,         ///< Color texture
        Roughness,      ///< PBR roughness
        Metallic,       ///< PBR metallic
        AO              ///< Ambient occlusion
    };

    // ============================================================================
    // PIN VALUE TYPES
    // ============================================================================
    
    /// 2D float array for heightmaps, masks, textures
    using ImageData = std::shared_ptr<std::vector<float>>;
    
    /// Image with dimensions
    struct Image2DData {
        ImageData data;
        int width = 0;
        int height = 0;
        int channels = 1;
        ImageSemantic semantic = ImageSemantic::Generic;
        
        bool isValid() const { return data && !data->empty() && width > 0 && height > 0; }
        size_t pixelCount() const { return static_cast<size_t>(width) * height; }
    };

    /**
     * @brief Type-safe variant holding any pin value
     * 
     * Uses std::monostate for empty/unconnected pins.
     * This is the core data container for all node I/O.
     */
    using PinValue = std::variant<
        std::monostate,             // Empty / None
        float,                      // Float
        int,                        // Int
        bool,                       // Bool
        std::array<float, 2>,       // Vector2
        std::array<float, 3>,       // Vector3
        std::array<float, 4>,       // Vector4 / Color
        Image2DData,                // Image2D
        std::string                 // String
    >;

    // ============================================================================
    // PIN SHAPE & VISUAL
    // ============================================================================
    
    enum class PinShape : uint8_t {
        Circle = 0,     ///< Default, most data types
        Square,         ///< Execution/control flow
        Diamond,        ///< Special connections (masks)
        Arrow           ///< Directional data
    };

    /**
     * @brief Visual configuration for a data type
     */
    struct DataTypeVisual {
        ImU32 color;
        PinShape shape;
        const char* displayName;
    };

    /**
     * @brief Get visual configuration for a data type
     */
    inline DataTypeVisual getDataTypeVisual(DataType type, ImageSemantic semantic = ImageSemantic::Generic) {
        switch (type) {
            case DataType::Float:
                return { IM_COL32(100, 180, 255, 255), PinShape::Circle, "Float" };
            case DataType::Int:
                return { IM_COL32(80, 200, 180, 255), PinShape::Circle, "Int" };
            case DataType::Bool:
                return { IM_COL32(200, 80, 80, 255), PinShape::Square, "Bool" };
            case DataType::Vector2:
                return { IM_COL32(255, 200, 100, 255), PinShape::Circle, "Vector2" };
            case DataType::Vector3:
                return { IM_COL32(255, 180, 50, 255), PinShape::Circle, "Vector3" };
            case DataType::Vector4:
            case DataType::Color:
                return { IM_COL32(255, 150, 200, 255), PinShape::Circle, "Color" };
            case DataType::Image2D:
                // Different colors based on semantic
                switch (semantic) {
                    case ImageSemantic::Height:
                        return { IM_COL32(100, 200, 100, 255), PinShape::Circle, "Height" };
                    case ImageSemantic::Mask:
                        return { IM_COL32(180, 100, 200, 255), PinShape::Diamond, "Mask" };
                    case ImageSemantic::Normal:
                        return { IM_COL32(130, 130, 255, 255), PinShape::Circle, "Normal" };
                    case ImageSemantic::Albedo:
                        return { IM_COL32(255, 180, 180, 255), PinShape::Circle, "Albedo" };
                    default:
                        return { IM_COL32(150, 150, 150, 255), PinShape::Circle, "Image" };
                }
            case DataType::String:
                return { IM_COL32(200, 200, 100, 255), PinShape::Circle, "String" };
            default:
                return { IM_COL32(128, 128, 128, 255), PinShape::Circle, "Unknown" };
        }
    }

    // ============================================================================
    // PIN KIND
    // ============================================================================
    
    enum class PinKind : uint8_t {
        Input = 0,
        Output = 1
    };

    // ============================================================================
    // ENHANCED PIN STRUCTURE
    // ============================================================================
    
    /**
     * @brief A connection point on a node
     * 
     * Pins are typed and can carry data or be connected to other pins.
     * The new design includes tooltips, optional connections, and default values.
     */
    struct Pin {
        // Identity
        uint32_t id = 0;
        uint32_t nodeId = 0;
        std::string name;
        std::string tooltip;            ///< Hover help text
        
        // Type information
        PinKind kind = PinKind::Input;
        DataType dataType = DataType::None;
        ImageSemantic imageSemantic = ImageSemantic::Generic;  ///< For Image2D pins
        
        // Connection rules
        bool allowMultipleConnections = false;  ///< Allow multiple inputs (for blend nodes)
        bool optional = false;                  ///< Can remain unconnected without error
        
        // Values
        PinValue defaultValue;          ///< Fallback when unconnected
        PinValue currentValue;          ///< Runtime computed value
        
        // Visual (cached from DataType)
        ImU32 cachedColor = 0;
        PinShape cachedShape = PinShape::Circle;
        
        // Methods
        void updateVisualCache() {
            auto visual = getDataTypeVisual(dataType, imageSemantic);
            cachedColor = visual.color;
            cachedShape = visual.shape;
        }
        
        bool canConnectTo(const Pin& other) const {
            // Basic rules: input->output or output->input, not same node
            if (kind == other.kind) return false;
            if (nodeId == other.nodeId) return false;
            
            // Type compatibility
            if (dataType == other.dataType) return true;
            
            // Allow some conversions
            // Float <-> Int
            if ((dataType == DataType::Float && other.dataType == DataType::Int) ||
                (dataType == DataType::Int && other.dataType == DataType::Float)) {
                return true;
            }
            
            // Image2D with different semantics can connect (height -> mask input)
            if (dataType == DataType::Image2D && other.dataType == DataType::Image2D) {
                return true;
            }
            
            return false;
        }
        
        bool hasValue() const {
            return !std::holds_alternative<std::monostate>(currentValue);
        }
        
        bool hasDefaultValue() const {
            return !std::holds_alternative<std::monostate>(defaultValue);
        }
        
        // ========================================================
        // FACTORY METHODS
        // ========================================================
        
        /**
         * @brief Create an input pin with type and optional flag
         */
        static Pin createInput(const std::string& name, DataType type, 
                               ImageSemantic semantic = ImageSemantic::Generic,
                               bool isOptional = false) {
            Pin pin;
            pin.name = name;
            pin.kind = PinKind::Input;
            pin.dataType = type;
            pin.imageSemantic = semantic;
            pin.optional = isOptional;
            pin.updateVisualCache();
            return pin;
        }
        
        /**
         * @brief Create an output pin
         */
        static Pin createOutput(const std::string& name, DataType type,
                                ImageSemantic semantic = ImageSemantic::Generic) {
            Pin pin;
            pin.name = name;
            pin.kind = PinKind::Output;
            pin.dataType = type;
            pin.imageSemantic = semantic;
            pin.updateVisualCache();
            return pin;
        }
    };

    // ============================================================================
    // LINK STRUCTURE
    // ============================================================================
    
    /**
     * @brief A connection between two pins
     */
    struct Link {
        uint32_t id = 0;
        uint32_t startPinId = 0;    ///< Output pin
        uint32_t endPinId = 0;      ///< Input pin
        
        // Visual options
        ImU32 colorOverride = 0;    ///< 0 = use pin color
        float thickness = 2.0f;
    };

    // ============================================================================
    // NODE METADATA
    // ============================================================================
    
    /**
     * @brief Rich metadata for node display and organization
     */
    struct NodeMetadata {
        std::string displayName;        ///< User-facing name
        std::string description;        ///< Detailed description
        std::string category;           ///< Primary category ("Erosion", "Noise", "Math")
        std::string subcategory;        ///< Secondary grouping
        std::vector<std::string> tags;  ///< Searchable keywords
        
        const char* iconName = nullptr; ///< Icon identifier (optional)
        std::string helpUrl;            ///< Link to documentation
        
        ImU32 headerColor = IM_COL32(60, 80, 100, 255);
        
        // Factory info
        std::string typeId;             ///< Unique type identifier for serialization
    };

    // ============================================================================
    // TYPE CONVERSION UTILITIES
    // ============================================================================
    
    /**
     * @brief Try to extract a float from a PinValue
     */
    inline bool tryGetFloat(const PinValue& value, float& out) {
        if (auto* f = std::get_if<float>(&value)) { out = *f; return true; }
        if (auto* i = std::get_if<int>(&value)) { out = static_cast<float>(*i); return true; }
        return false;
    }

    /**
     * @brief Try to extract an int from a PinValue
     */
    inline bool tryGetInt(const PinValue& value, int& out) {
        if (auto* i = std::get_if<int>(&value)) { out = *i; return true; }
        if (auto* f = std::get_if<float>(&value)) { out = static_cast<int>(*f); return true; }
        return false;
    }

    /**
     * @brief Try to extract Image2D data from a PinValue
     */
    inline bool tryGetImage(const PinValue& value, Image2DData& out) {
        if (auto* img = std::get_if<Image2DData>(&value)) { 
            out = *img; 
            return img->isValid(); 
        }
        return false;
    }

    /**
     * @brief Create an Image2D PinValue from raw float data
     */
    inline PinValue makeImageValue(const std::vector<float>& data, int width, int height, 
                                    int channels = 1, ImageSemantic semantic = ImageSemantic::Generic) {
        Image2DData img;
        img.data = std::make_shared<std::vector<float>>(data);
        img.width = width;
        img.height = height;
        img.channels = channels;
        img.semantic = semantic;
        return img;
    }

    /**
     * @brief Create an Image2D PinValue by wrapping existing shared data
     */
    inline PinValue wrapImageValue(ImageData data, int width, int height,
                                    int channels = 1, ImageSemantic semantic = ImageSemantic::Generic) {
        Image2DData img;
        img.data = data;
        img.width = width;
        img.height = height;
        img.channels = channels;
        img.semantic = semantic;
        return img;
    }

} // namespace NodeSystem

