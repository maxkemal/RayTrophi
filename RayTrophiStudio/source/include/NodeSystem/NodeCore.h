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
#include <limits>
#include "imgui.h"

// Forward declaration only (not a full include) — DataType::Geometry's PinValue payload is
// std::shared_ptr<TriangleMesh>, which is legal to hold in a std::variant/shared_ptr member
// with an incomplete type (the deleter is captured at construction). Keeps NodeCore.h — included
// by every node-system consumer — free of the Hittable/DNA::GeometryDetail include chain.
class TriangleMesh;

// Same forward-declare pattern for DataType::Material's payload: a CPU-side snapshot of a full
// material parameter set + texture bindings (defined in MaterialNodesV2.h). Keeps NodeCore.h
// free of the Material/Texture include chain.
namespace MaterialNodesV2 { struct ShadeState; }

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
        Geometry,       ///< GeometryDetail SoA mesh/spline
        Instances,      ///< Scatter/Instance transform arrays
        Volume,         ///< Volume grid (NanoVDB/OpenVDB)
        Simulation,     ///< Simulation state (cache-backed)
        Material,       ///< Material shader DAG handle
        Light,          ///< Light parameters variant
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
        AO,             ///< Ambient occlusion
        // Appended for serialized enum stability. These distinguish fields
        // that share Image2D storage but must not be wired interchangeably.
        PhysicalScalar, ///< SI-valued scalar field (depth, area, discharge...)
        Direction,      ///< Encoded direction/vector field
        Categorical,    ///< Discrete IDs/classes/orders
        PackedData      ///< Multi-channel packed data
    };

    enum class ImageUnit : uint8_t {
        Unknown = 0,        ///< Legacy/wildcard unit
        Unitless,           ///< Ratios and normalized values
        Meters,
        SquareMeters,
        CubicMetersPerSecond,
        MetersPerSecond,
        MillimetersPerHour,
        Degrees,
        Identifier
    };

    inline const char* getImageUnitName(ImageUnit unit) {
        switch (unit) {
            case ImageUnit::Unitless: return "unitless";
            case ImageUnit::Meters: return "m";
            case ImageUnit::SquareMeters: return "m2";
            case ImageUnit::CubicMetersPerSecond: return "m3/s";
            case ImageUnit::MetersPerSecond: return "m/s";
            case ImageUnit::MillimetersPerHour: return "mm/h";
            case ImageUnit::Degrees: return "deg";
            case ImageUnit::Identifier: return "ID";
            default: return "";
        }
    }

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
        ImageUnit unit = ImageUnit::Unknown;
        
        bool isValid() const {
            if (!data || width <= 0 || height <= 0 || channels <= 0) return false;
            const size_t w = static_cast<size_t>(width);
            const size_t h = static_cast<size_t>(height);
            const size_t c = static_cast<size_t>(channels);
            if (w > (std::numeric_limits<size_t>::max)() / h) return false;
            const size_t pixels = w * h;
            if (pixels > (std::numeric_limits<size_t>::max)() / c) return false;
            return data->size() == pixels * c;
        }
        size_t pixelCount() const { return static_cast<size_t>(width) * height; }
    };

    /**
     * @brief Type-safe variant holding any pin value
     * 
     * Uses std::monostate for empty/unconnected pins.
     * This is the core data container for all node I/O.
     */
    /// Geometry payload: reuses the same TriangleMesh (flat SoA / DNA::GeometryDetail) type
    /// every other consumer in the codebase already knows how to handle (push directly into
    /// scene.world.objects, hand to Vulkan/OptiX/Embree BLAS builders, etc.) — no new wrapper
    /// type, and sharing the shared_ptr between DAG nodes is inherently zero-copy.
    using GeometryValue = std::shared_ptr<TriangleMesh>;

    /// Material payload: full shading parameter set + texture bindings snapshot
    /// (MaterialNodesV2::ShadeState). Shared between MaterialRef / MixMaterial /
    /// Output nodes by shared_ptr — zero-copy through the graph.
    using MaterialValue = std::shared_ptr<MaterialNodesV2::ShadeState>;

    using PinValue = std::variant<
        std::monostate,             // Empty / None
        float,                      // Float
        int,                        // Int
        bool,                       // Bool
        std::array<float, 2>,       // Vector2
        std::array<float, 3>,       // Vector3
        std::array<float, 4>,       // Vector4 / Color
        Image2DData,                // Image2D
        std::string,                // String
        GeometryValue,              // Geometry (DataType::Geometry)
        MaterialValue               // Material (DataType::Material)
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
                    case ImageSemantic::PhysicalScalar:
                        return { IM_COL32(65, 180, 225, 255), PinShape::Circle, "Physical Scalar" };
                    case ImageSemantic::Direction:
                        return { IM_COL32(45, 210, 195, 255), PinShape::Arrow, "Direction" };
                    case ImageSemantic::Categorical:
                        return { IM_COL32(235, 185, 65, 255), PinShape::Square, "Categorical" };
                    case ImageSemantic::PackedData:
                        return { IM_COL32(235, 105, 165, 255), PinShape::Circle, "Packed Data" };
                    default:
                        return { IM_COL32(150, 150, 150, 255), PinShape::Circle, "Image" };
                }
            case DataType::String:
                return { IM_COL32(200, 200, 100, 255), PinShape::Circle, "String" };
            case DataType::Geometry:
                return { IM_COL32(76, 175, 80, 255), PinShape::Circle, "Geometry" };
            case DataType::Instances:
                return { IM_COL32(255, 235, 59, 255), PinShape::Circle, "Instances" };
            case DataType::Volume:
                return { IM_COL32(156, 39, 176, 255), PinShape::Circle, "Volume" };
            case DataType::Simulation:
                return { IM_COL32(255, 87, 34, 255), PinShape::Square, "Simulation" };
            case DataType::Material:
                return { IM_COL32(233, 30, 99, 255), PinShape::Circle, "Material" };
            case DataType::Light:
                return { IM_COL32(255, 193, 7, 255), PinShape::Circle, "Light" };
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
        int imageChannels = 1;                                 ///< 0 accepts any channel count
        ImageUnit imageUnit = ImageUnit::Unknown;               ///< Physical unit/wildcard
        uint32_t acceptedImageSemantics = 0;                    ///< Extra accepted input semantics
        
        // Connection rules
        bool allowMultipleConnections = false;  ///< Allow multiple inputs (for blend nodes)
        bool optional = false;                  ///< Can remain unconnected without error

        // Editor-only: pin belongs to a collapsed group on the node (e.g. the
        // Material Output's optional socket groups). A hidden pin is not drawn
        // and not registered for interaction (pinPositions_ skips it), so it
        // can't be link-targeted. Callers must never hide a CONNECTED pin —
        // NodeEditorUIV2 draws links from cached pin positions.
        bool hidden = false;
        
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

        void acceptImageSemantic(ImageSemantic semantic) {
            const uint32_t value = static_cast<uint32_t>(semantic);
            if (value < 32u) acceptedImageSemantics |= (1u << value);
        }

        bool acceptsImageSemantic(ImageSemantic semantic) const {
            if (imageSemantic == ImageSemantic::Generic || semantic == ImageSemantic::Generic ||
                imageSemantic == semantic) return true;
            const uint32_t value = static_cast<uint32_t>(semantic);
            return value < 32u && (acceptedImageSemantics & (1u << value)) != 0u;
        }
        
        bool canConnectTo(const Pin& other) const {
            // Basic rules: input->output or output->input, not same node
            if (kind == other.kind) return false;
            if (nodeId == other.nodeId) return false;
            
            if (dataType == DataType::Image2D && other.dataType == DataType::Image2D) {
                if (imageChannels > 0 && other.imageChannels > 0 &&
                    imageChannels != other.imageChannels) return false;
                const Pin& inputPin = kind == PinKind::Input ? *this : other;
                const Pin& outputPin = kind == PinKind::Output ? *this : other;
                const bool semanticCompatible = inputPin.acceptsImageSemantic(outputPin.imageSemantic);
                if (!semanticCompatible) return false;
                const bool unitCompatible = imageUnit == other.imageUnit ||
                    imageUnit == ImageUnit::Unknown || other.imageUnit == ImageUnit::Unknown;
                return unitCompatible;
            }

            // Type compatibility
            if (dataType == other.dataType) return true;
            
            // Allow some conversions
            // Float <-> Int
            if ((dataType == DataType::Float && other.dataType == DataType::Int) ||
                (dataType == DataType::Int && other.dataType == DataType::Float)) {
                return true;
            }

            // Float <-> Vector3 (material graphs: Noise Fac -> Base Color,
            // Image Texture Color -> Roughness, ...). Consumers convert via
            // splat / channel-average helpers.
            if ((dataType == DataType::Float && other.dataType == DataType::Vector3) ||
                (dataType == DataType::Vector3 && other.dataType == DataType::Float)) {
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
                               bool isOptional = false, int channels = 1,
                               ImageUnit unit = ImageUnit::Unknown) {
            Pin pin;
            pin.name = name;
            pin.kind = PinKind::Input;
            pin.dataType = type;
            pin.imageSemantic = semantic;
            pin.imageChannels = channels;
            pin.imageUnit = unit;
            pin.optional = isOptional;
            pin.updateVisualCache();
            return pin;
        }
        
        /**
         * @brief Create an output pin
         */
        static Pin createOutput(const std::string& name, DataType type,
                                ImageSemantic semantic = ImageSemantic::Generic,
                                int channels = 1,
                                ImageUnit unit = ImageUnit::Unknown) {
            Pin pin;
            pin.name = name;
            pin.kind = PinKind::Output;
            pin.dataType = type;
            pin.imageSemantic = semantic;
            pin.imageChannels = channels;
            pin.imageUnit = unit;
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
     * @brief Try to extract a Geometry (TriangleMesh) value from a PinValue
     */
    inline bool tryGetGeometry(const PinValue& value, GeometryValue& out) {
        if (auto* mesh = std::get_if<GeometryValue>(&value)) {
            out = *mesh;
            return static_cast<bool>(*mesh);
        }
        return false;
    }

    /**
     * @brief Try to extract a Material (ShadeState) value from a PinValue
     */
    inline bool tryGetMaterial(const PinValue& value, MaterialValue& out) {
        if (auto* mat = std::get_if<MaterialValue>(&value)) {
            out = *mat;
            return static_cast<bool>(*mat);
        }
        return false;
    }

    /**
     * @brief Create an Image2D PinValue from raw float data
     */
    inline PinValue makeImageValue(const std::vector<float>& data, int width, int height, 
                                    int channels = 1, ImageSemantic semantic = ImageSemantic::Generic,
                                    ImageUnit unit = ImageUnit::Unknown) {
        Image2DData img;
        img.data = std::make_shared<std::vector<float>>(data);
        img.width = width;
        img.height = height;
        img.channels = channels;
        img.semantic = semantic;
        img.unit = unit;
        return img;
    }

    /**
     * @brief Create an Image2D PinValue by wrapping existing shared data
     */
    inline PinValue wrapImageValue(ImageData data, int width, int height,
                                    int channels = 1, ImageSemantic semantic = ImageSemantic::Generic,
                                    ImageUnit unit = ImageUnit::Unknown) {
        Image2DData img;
        img.data = data;
        img.width = width;
        img.height = height;
        img.channels = channels;
        img.semantic = semantic;
        img.unit = unit;
        return img;
    }

} // namespace NodeSystem

