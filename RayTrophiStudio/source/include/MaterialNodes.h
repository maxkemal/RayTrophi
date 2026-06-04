/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          MaterialNodes.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once

/**
 * @file MaterialNodes.h
 * @brief Procedural Material Node System
 * 
 * Implements a node graph for defining material properties (Albedo, Roughness, etc.)
 * using the generic NodeSystem. Features texture loading, math operations, and
 * baking support.
 */

#include "NodeSystem/NodeCore.h"
#include "NodeSystem/Node.h"
#include "NodeSystem/Graph.h"
#include "NodeSystem/EvaluationContext.h"
#include "stb_image.h"
#include "json.hpp"
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

// ============================================================================
// INPUT NODES
// ============================================================================


namespace MaterialNodes {

    // ============================================================================
    // CONTEXT
    // ============================================================================
    
    struct MaterialContext {
        // resolution for baking
        int width = 1024;
        int height = 1024;
        
        // Current UV being evaluated (for future per-pixel usage)
        float u = 0.0f;
        float v = 0.0f;
    };

    // ============================================================================
    // NODE TYPES
    // ============================================================================
    
    enum class NodeType {
        // Output
        MaterialOutput,
        
        // Input
        Value,
        RGB,
        TextureCoordinate,
        
        // Texture
        ImageTexture,
        NoiseTexture, // Perlin
        CheckerTexture,
        
        // Color
        MixRGB,
        ColorRamp,
        Invert,
        Gamma,
        
        // Vector
        Mapping,
        
        // Math
        Math
    };

    // ============================================================================
    // BASE CLASS
    // ============================================================================
    
    using Vec3Data = std::array<float, 3>;
    using FloatData = float;

    class MaterialNodeBase : public NodeSystem::NodeBase {
    public:
        NodeType matNodeType;
        
        // Helper to get Color (Vec3) input
        std::array<float, 3> getColorInput(int index, NodeSystem::EvaluationContext& ctx) {
            NodeSystem::PinValue val = getInputValue(index, ctx);
            if (auto* v = std::get_if<std::array<float, 3>>(&val)) return *v;
            if (auto* f = std::get_if<float>(&val)) return std::array<float, 3>{*f, *f, *f};
            return std::array<float, 3>{0,0,0};
        }
        
        // Helper to get Float input
        float getFloatInput(int index, NodeSystem::EvaluationContext& ctx) {
            NodeSystem::PinValue val = getInputValue(index, ctx);
            if (auto* f = std::get_if<float>(&val)) return *f;
            if (auto* v = std::get_if<std::array<float, 3>>(&val)) return ((*v)[0] + (*v)[1] + (*v)[2]) / 3.0f;
            return 0.0f;
        }

        // Serialization
        virtual void serializeToJson(nlohmann::json& j) const {
            j["x"] = x;
            j["y"] = y;
            j["type"] = static_cast<int>(matNodeType);
            j["name"] = name;
        }

        virtual void deserializeFromJson(const nlohmann::json& j) {
            if (j.contains("x")) x = j["x"].get<float>();
            if (j.contains("y")) y = j["y"].get<float>();
            if (j.contains("name")) name = j["name"].get<std::string>();
        }
    };

    // ... (Updating Output Node to use std::array) ...

    struct BakeResult {
        int width, height;
        // All channels are normalized 0-255 or RGB
        std::vector<unsigned char> albedo;    // RGBA
        std::vector<unsigned char> metallic;  // R
        std::vector<unsigned char> roughness; // R
        std::vector<unsigned char> emission;  // RGB
        std::vector<unsigned char> normal;    // RGB
        
        // Extended PBR
        std::vector<unsigned char> transmission; // R
        std::vector<unsigned char> ior;          // R (remapped?) or just raw float baked separately? let's stick to 0-1 for now, or just not bake IOR often.
                                                 // Actually IOR > 1.0. Baking IOR to texture is rare. Let's keep it simple.
                                                 // But user wants "PrincipledBSDF exact channels".
                                                 // Let's store IOR as 0.0-10.0 mapped to 0-255? Or just support it later.
                                                 // For now, let's add standard visual ones.
        std::vector<unsigned char> alpha;        // R
        std::vector<unsigned char> specular;     // R
        std::vector<unsigned char> sss_color;    // RGB
        std::vector<unsigned char> sss_scale;    // R
        std::vector<unsigned char> clearcoat;    // R
        std::vector<unsigned char> clearcoat_roughness; // R
        std::vector<unsigned char> anisotropic;  // R
        // std::vector<unsigned char> sheen;     // R (if needed)
    };

    inline BakeResult bakeGraph(MaterialNodeGraph& graph, int w, int h); // Fwd decl

    // ...

     class MaterialOutputNode : public MaterialNodeBase {
    public:
        MaterialOutputNode() {
            name = "Material Output";
            matNodeType = NodeType::MaterialOutput;
            
            // Standard PBR inputs matching PrincipledBSDF
            inputs.push_back(NodeSystem::Pin::createInput("Base Color", NodeSystem::DataType::Vector3)); // 0
            inputs.push_back(NodeSystem::Pin::createInput("Metallic", NodeSystem::DataType::Float));     // 1
            inputs.push_back(NodeSystem::Pin::createInput("Roughness", NodeSystem::DataType::Float));    // 2
            inputs.push_back(NodeSystem::Pin::createInput("Emission", NodeSystem::DataType::Vector3));   // 3
            inputs.push_back(NodeSystem::Pin::createInput("Normal", NodeSystem::DataType::Vector3));     // 4
            
            // Extended
            inputs.push_back(NodeSystem::Pin::createInput("Alpha", NodeSystem::DataType::Float));        // 5 (Opacity)
            inputs.back().defaultValue = 1.0f;
            
            inputs.push_back(NodeSystem::Pin::createInput("Transmission", NodeSystem::DataType::Float)); // 6
            inputs.push_back(NodeSystem::Pin::createInput("IOR", NodeSystem::DataType::Float));          // 7
            inputs.back().defaultValue = 1.45f;
            
            inputs.push_back(NodeSystem::Pin::createInput("Specular", NodeSystem::DataType::Float));     // 8
            inputs.back().defaultValue = 0.5f;

            inputs.push_back(NodeSystem::Pin::createInput("SSS Color", NodeSystem::DataType::Vector3));  // 9
            inputs.push_back(NodeSystem::Pin::createInput("SSS Scale", NodeSystem::DataType::Float));    // 10
            
            inputs.push_back(NodeSystem::Pin::createInput("Clearcoat", NodeSystem::DataType::Float));           // 11
            inputs.push_back(NodeSystem::Pin::createInput("Clearcoat Roughness", NodeSystem::DataType::Float)); // 12
            
            inputs.push_back(NodeSystem::Pin::createInput("Anisotropic", NodeSystem::DataType::Float));  // 13
            
            headerColor = ImVec4(0.2f, 0.2f, 0.2f, 1.0f);
        }
        
        // Output node doesn't compute values for itself, it is pulled by the baker
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override {
            return NodeSystem::PinValue{};
        }
        
        std::string getTypeId() const override { return "Mat.Output"; }
    };

    // ... (ImageTextureNode) ...
    class ImageTextureNode : public MaterialNodeBase {
    public:
        std::string filePath;
        int width=0, height=0, channels=0;
        std::vector<unsigned char> rawData;
        bool isLoaded = false;
        bool browseForFile = false; // Flag for UI
        
        ImageTextureNode() {
            name = "Image Texture";
            matNodeType = NodeType::ImageTexture;
            
            inputs.push_back(NodeSystem::Pin::createInput("Vector", NodeSystem::DataType::Vector3)); // UV
            
            outputs.push_back(NodeSystem::Pin::createOutput("Color", NodeSystem::DataType::Vector3));
            outputs.push_back(NodeSystem::Pin::createOutput("R", NodeSystem::DataType::Float));
            outputs.push_back(NodeSystem::Pin::createOutput("G", NodeSystem::DataType::Float));
            outputs.push_back(NodeSystem::Pin::createOutput("B", NodeSystem::DataType::Float));
            outputs.push_back(NodeSystem::Pin::createOutput("Alpha", NodeSystem::DataType::Float));
            
            headerColor = ImVec4(0.7f, 0.4f, 0.2f, 1.0f);
        }

        // ... loadFile same ...
        void loadFile() {
             if (filePath.empty()) return;
             // if (isLoaded) return; // Allow reload?
             
             unsigned char* img = stbi_load(filePath.c_str(), &width, &height, &channels, 4); // Force RGBA
             if (img) {
                 rawData.assign(img, img + (width * height * 4));
                 stbi_image_free(img);
                 isLoaded = true;
             }
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override {
            if (!isLoaded || width == 0 || height == 0) return std::array<float, 3>{0.8f, 0.0f, 0.8f}; // Missing texture magenta
            
            // Get UV from input or context
            std::array<float, 3> uv = {0,0,0};
            
            auto* matCtx = ctx.getDomainContext<MaterialContext>();
            if (matCtx) {
                // Default UV from context (current pixel/vertex)
                uv = {matCtx->u, matCtx->v, 0}; 
            }
            
            // If Vector input is connected (returns non-monostate), override UV
            NodeSystem::PinValue vecVal = getInputValue(0, ctx);
            if (!std::holds_alternative<std::monostate>(vecVal)) {
                 if (auto* v = std::get_if<std::array<float, 3>>(&vecVal)) uv = *v;
                 else if (auto* f = std::get_if<float>(&vecVal)) uv = {*f, *f, *f};
            }
            
            // Wrap UV
            float u = uv[0] - floor(uv[0]);
            float v = uv[1] - floor(uv[1]);
            
            // Bilinear Sample
            float tx = u * (width-1);
            float ty = v * (height-1);
            int x0 = (int)tx;
            int y0 = (int)ty;
            int x1 = std::min(x0+1, width-1);
            int y1 = std::min(y0+1, height-1);
            
            float fx = tx - x0;
            float fy = ty - y0;
            
            auto getPx = [&](int x, int y) {
                int idx = (y * width + x) * 4;
                return ImVec4(rawData[idx]/255.0f, rawData[idx+1]/255.0f, rawData[idx+2]/255.0f, rawData[idx+3]/255.0f);
            };
            
            ImVec4 c00 = getPx(x0, y0);
            ImVec4 c10 = getPx(x1, y0);
            ImVec4 c01 = getPx(x0, y1);
            ImVec4 c11 = getPx(x1, y1);
            
            // Interpolate
            auto lerp = [](ImVec4 a, ImVec4 b, float t) {
                return ImVec4(a.x+(b.x-a.x)*t, a.y+(b.y-a.y)*t, a.z+(b.z-a.z)*t, a.w+(b.w-a.w)*t);
            };
            
            ImVec4 top = lerp(c00, c10, fx);
            ImVec4 bot = lerp(c01, c11, fx);
            ImVec4 f = lerp(top, bot, fy); // rename final to f prevents keyword clash
            
            if (outputIndex == 0) return std::array<float, 3>{f.x, f.y, f.z};
            if (outputIndex == 1) return f.x; // R
            if (outputIndex == 2) return f.y; // G
            if (outputIndex == 3) return f.z; // B
            if (outputIndex == 4) return f.w; // Alpha
            
            return NodeSystem::PinValue{};
        }

        // ... (drawContent serialization etc) ...
        void drawContent() override {
            if (ImGui::Button("Open...")) {
                browseForFile = true;
            }
            ImGui::SameLine();
            if (isLoaded) {
                 ImGui::Text("%dx%d", width, height);
            } else {
                 ImGui::Text("No Image");
            }
        }
        
        void serializeToJson(nlohmann::json& j) const override {
            MaterialNodeBase::serializeToJson(j);
            j["filePath"] = filePath;
        }
        
        void deserializeFromJson(const nlohmann::json& j) override {
            MaterialNodeBase::deserializeFromJson(j);
            if (j.contains("filePath")) {
                filePath = j["filePath"].get<std::string>();
                loadFile();
            }
        }
        
        std::string getTypeId() const override { return "Mat.ImageTexture"; }
        float getCustomWidth() const override { return 160.0f; }
    };
    
    
    class ValueNode : public MaterialNodeBase {
    public:
        float value = 0.5f;

        ValueNode() {
            name = "Value";
            matNodeType = NodeType::Value;
            outputs.push_back(NodeSystem::Pin::createOutput("Val", NodeSystem::DataType::Float));
            headerColor = ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
        }

        NodeSystem::PinValue compute(int, NodeSystem::EvaluationContext&) override {
            return value;
        }

        void drawContent() override {
            if (ImGui::DragFloat("##v", &value, 0.01f)) {
                dirty = true;
            }
        }

        void serializeToJson(nlohmann::json& j) const override {
            MaterialNodeBase::serializeToJson(j);
            j["value"] = value;
        }

        void deserializeFromJson(const nlohmann::json& j) override {
            MaterialNodeBase::deserializeFromJson(j);
            if (j.contains("value")) value = j["value"];
        }

        std::string getTypeId() const override { return "Mat.Value"; }
    };
    
    class TextureCoordinateNode : public MaterialNodeBase {
    public:
        TextureCoordinateNode() {
             name = "Texture Coordinate";
             matNodeType = NodeType::TextureCoordinate;
             
             outputs.push_back(NodeSystem::Pin::createOutput("UV", NodeSystem::DataType::Vector3));
             outputs.push_back(NodeSystem::Pin::createOutput("Normal", NodeSystem::DataType::Vector3));
             
             headerColor = ImVec4(0.8f, 0.2f, 0.2f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override {
            auto* matCtx = ctx.getDomainContext<MaterialContext>();
            if (!matCtx) return std::array<float, 3>{0,0,0};
            
            if (outputIndex == 0) return std::array<float, 3>{matCtx->u, matCtx->v, 0}; // UV
            // Normal would require advanced context, return Up for now
            if (outputIndex == 1) return std::array<float, 3>{0, 1, 0}; 
            
            return NodeSystem::PinValue{};
        }
        std::string getTypeId() const override { return "Mat.TexCoord"; }
    };

    class RGBNode : public MaterialNodeBase {
    public:
        ImVec4 color = ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
        
        RGBNode() {
            name = "RGB";
            matNodeType = NodeType::RGB;
            outputs.push_back(NodeSystem::Pin::createOutput("Color", NodeSystem::DataType::Vector3));
            headerColor = ImVec4(0.8f, 0.8f, 0.0f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int, NodeSystem::EvaluationContext&) override {
            return std::array<float, 3>{color.x, color.y, color.z};
        }
        
        void drawContent() override {
            if (ImGui::ColorEdit3("Val", (float*)&color, ImGuiColorEditFlags_NoInputs)) {
                dirty = true;
            }
        }
        
        void serializeToJson(nlohmann::json& j) const override {
            MaterialNodeBase::serializeToJson(j);
            j["r"] = color.x; j["g"] = color.y; j["b"] = color.z;
        }
        
        void deserializeFromJson(const nlohmann::json& j) override {
            MaterialNodeBase::deserializeFromJson(j);
            if(j.contains("r")) color.x = j["r"];
            if(j.contains("g")) color.y = j["g"];
            if(j.contains("b")) color.z = j["b"];
        }
        
        std::string getTypeId() const override { return "Mat.RGB"; }
    };

     class MixRGBNode : public MaterialNodeBase {
    public:
         MixRGBNode() {
             name = "Mix RGB";
             matNodeType = NodeType::MixRGB;
             
             inputs.push_back(NodeSystem::Pin::createInput("Fac", NodeSystem::DataType::Float));
             inputs.back().defaultValue = 0.5f;
             
             inputs.push_back(NodeSystem::Pin::createInput("Color1", NodeSystem::DataType::Vector3));
             inputs.back().defaultValue = std::array<float, 3>{0.5f, 0.5f, 0.5f};
             
             inputs.push_back(NodeSystem::Pin::createInput("Color2", NodeSystem::DataType::Vector3));
             inputs.back().defaultValue = std::array<float, 3>{0.5f, 0.5f, 0.5f};
             
             outputs.push_back(NodeSystem::Pin::createOutput("Color", NodeSystem::DataType::Vector3));
             headerColor = ImVec4(0.8f, 0.8f, 0.0f, 1.0f);
         }
         
         NodeSystem::PinValue compute(int, NodeSystem::EvaluationContext& ctx) override {
             float fac = getFloatInput(0, ctx);
             std::array<float, 3> c1 = getColorInput(1, ctx);
             std::array<float, 3> c2 = getColorInput(2, ctx);
             
             // Clamp fac
             fac = std::max(0.0f, std::min(1.0f, fac));
             
             return std::array<float, 3>{
                 c1[0] * (1 - fac) + c2[0] * fac,
                 c1[1] * (1 - fac) + c2[1] * fac,
                 c1[2] * (1 - fac) + c2[2] * fac
             };
         }
         std::string getTypeId() const override { return "Mat.MixRGB"; }
    };

    // ============================================================================
    // NOISE TEXTURE (Perlin Noise)
    // ============================================================================
    class NoiseTextureNode : public MaterialNodeBase {
    public:
        float scale = 5.0f;
        float detail = 2.0f;
        
        NoiseTextureNode() {
            name = "Noise Texture";
            matNodeType = NodeType::NoiseTexture;
            
            inputs.push_back(NodeSystem::Pin::createInput("Vector", NodeSystem::DataType::Vector3));
            inputs.push_back(NodeSystem::Pin::createInput("Scale", NodeSystem::DataType::Float));
            inputs.back().defaultValue = 5.0f;
            
            outputs.push_back(NodeSystem::Pin::createOutput("Fac", NodeSystem::DataType::Float));
            outputs.push_back(NodeSystem::Pin::createOutput("Color", NodeSystem::DataType::Vector3));
            
            headerColor = ImVec4(0.6f, 0.3f, 0.6f, 1.0f);
        }
        
        // Simple Perlin-like noise
        static float fade(float t) { return t * t * t * (t * (t * 6 - 15) + 10); }
        static float lerp(float a, float b, float t) { return a + t * (b - a); }
        static float grad(int hash, float x, float y, float z) {
            int h = hash & 15;
            float u = h < 8 ? x : y;
            float v = h < 4 ? y : h == 12 || h == 14 ? x : z;
            return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
        }
        
        float noise3D(float x, float y, float z) const {
            static const int p[512] = {
                151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
                8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,
                117,35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,
                165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,
                105,92,41,55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,
                187,208,89,18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,
                64,52,217,226,250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,
                47,16,58,17,182,189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,
                153,101,155,167,43,172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,
                112,104,218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,
                145,235,249,14,239,107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,
                50,45,127,4,150,254,138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,
                215,61,156,180,
                151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
                8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,
                117,35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,
                165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,
                105,92,41,55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,
                187,208,89,18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,
                64,52,217,226,250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,
                47,16,58,17,182,189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,
                153,101,155,167,43,172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,
                112,104,218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,
                145,235,249,14,239,107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,
                50,45,127,4,150,254,138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,
                215,61,156,180
            };
            
            int X = (int)floor(x) & 255, Y = (int)floor(y) & 255, Z = (int)floor(z) & 255;
            x -= floor(x); y -= floor(y); z -= floor(z);
            float u = fade(x), v = fade(y), w = fade(z);
            
            int A = p[X] + Y, AA = p[A] + Z, AB = p[A+1] + Z;
            int B = p[X+1] + Y, BA = p[B] + Z, BB = p[B+1] + Z;
            
            return lerp(lerp(lerp(grad(p[AA], x, y, z), grad(p[BA], x-1, y, z), u),
                             lerp(grad(p[AB], x, y-1, z), grad(p[BB], x-1, y-1, z), u), v),
                        lerp(lerp(grad(p[AA+1], x, y, z-1), grad(p[BA+1], x-1, y, z-1), u),
                             lerp(grad(p[AB+1], x, y-1, z-1), grad(p[BB+1], x-1, y-1, z-1), u), v), w);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override {
            // Get UV
            std::array<float, 3> uv = {0,0,0};
            auto* matCtx = ctx.getDomainContext<MaterialContext>();
            if (matCtx) uv = {matCtx->u, matCtx->v, 0};
            
            NodeSystem::PinValue vecVal = getInputValue(0, ctx);
            if (!std::holds_alternative<std::monostate>(vecVal)) {
                if (auto* v = std::get_if<std::array<float, 3>>(&vecVal)) uv = *v;
            }
            
            float s = scale;
            NodeSystem::PinValue scaleVal = getInputValue(1, ctx);
            if (auto* f = std::get_if<float>(&scaleVal)) s = *f;
            
            float n = noise3D(uv[0] * s, uv[1] * s, uv[2] * s);
            n = n * 0.5f + 0.5f; // Map to 0-1
            
            if (outputIndex == 0) return n;
            return std::array<float, 3>{n, n, n};
        }
        
        void drawContent() override {
            if (ImGui::DragFloat("Scale", &scale, 0.1f, 0.1f, 100.0f)) dirty = true;
        }
        
        void serializeToJson(nlohmann::json& j) const override {
            MaterialNodeBase::serializeToJson(j);
            j["scale"] = scale;
        }
        
        void deserializeFromJson(const nlohmann::json& j) override {
            MaterialNodeBase::deserializeFromJson(j);
            if (j.contains("scale")) scale = j["scale"];
        }
        
        std::string getTypeId() const override { return "Mat.NoiseTexture"; }
    };

    // ============================================================================
    // MATH NODE
    // ============================================================================
    enum class MathOperation { Add, Subtract, Multiply, Divide, Power, Sqrt, Abs, Min, Max };
    
    class MathNode : public MaterialNodeBase {
    public:
        MathOperation operation = MathOperation::Add;
        
        MathNode() {
            name = "Math";
            matNodeType = NodeType::Math;
            
            inputs.push_back(NodeSystem::Pin::createInput("Value1", NodeSystem::DataType::Float));
            inputs.back().defaultValue = 0.5f;
            inputs.push_back(NodeSystem::Pin::createInput("Value2", NodeSystem::DataType::Float));
            inputs.back().defaultValue = 0.5f;
            
            outputs.push_back(NodeSystem::Pin::createOutput("Value", NodeSystem::DataType::Float));
            
            headerColor = ImVec4(0.4f, 0.4f, 0.7f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int, NodeSystem::EvaluationContext& ctx) override {
            float a = getFloatInput(0, ctx);
            float b = getFloatInput(1, ctx);
            
            switch (operation) {
                case MathOperation::Add:      return a + b;
                case MathOperation::Subtract: return a - b;
                case MathOperation::Multiply: return a * b;
                case MathOperation::Divide:   return b != 0 ? a / b : 0.0f;
                case MathOperation::Power:    return std::pow(a, b);
                case MathOperation::Sqrt:     return std::sqrt(std::max(0.0f, a));
                case MathOperation::Abs:      return std::abs(a);
                case MathOperation::Min:      return std::min(a, b);
                case MathOperation::Max:      return std::max(a, b);
            }
            return 0.0f;
        }
        
        void drawContent() override {
            const char* ops[] = {"Add", "Subtract", "Multiply", "Divide", "Power", "Sqrt", "Abs", "Min", "Max"};
            int current = static_cast<int>(operation);
            if (ImGui::Combo("##op", &current, ops, 9)) {
                operation = static_cast<MathOperation>(current);
                dirty = true;
            }
        }
        
        void serializeToJson(nlohmann::json& j) const override {
            MaterialNodeBase::serializeToJson(j);
            j["operation"] = static_cast<int>(operation);
        }
        
        void deserializeFromJson(const nlohmann::json& j) override {
            MaterialNodeBase::deserializeFromJson(j);
            if (j.contains("operation")) operation = static_cast<MathOperation>(j["operation"].get<int>());
        }
        
        std::string getTypeId() const override { return "Mat.Math"; }
    };

    // ============================================================================
    // SEPARATE RGB / COMBINE RGB
    // ============================================================================
    class SeparateRGBNode : public MaterialNodeBase {
    public:
        SeparateRGBNode() {
            name = "Separate RGB";
            matNodeType = NodeType::Invert; // Reuse type for now
            
            inputs.push_back(NodeSystem::Pin::createInput("Color", NodeSystem::DataType::Vector3));
            
            outputs.push_back(NodeSystem::Pin::createOutput("R", NodeSystem::DataType::Float));
            outputs.push_back(NodeSystem::Pin::createOutput("G", NodeSystem::DataType::Float));
            outputs.push_back(NodeSystem::Pin::createOutput("B", NodeSystem::DataType::Float));
            
            headerColor = ImVec4(0.3f, 0.6f, 0.3f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override {
            std::array<float, 3> c = getColorInput(0, ctx);
            if (outputIndex == 0) return c[0];
            if (outputIndex == 1) return c[1];
            if (outputIndex == 2) return c[2];
            return 0.0f;
        }
        
        std::string getTypeId() const override { return "Mat.SeparateRGB"; }
    };

    class CombineRGBNode : public MaterialNodeBase {
    public:
        CombineRGBNode() {
            name = "Combine RGB";
            matNodeType = NodeType::Gamma; // Reuse type for now
            
            inputs.push_back(NodeSystem::Pin::createInput("R", NodeSystem::DataType::Float));
            inputs.push_back(NodeSystem::Pin::createInput("G", NodeSystem::DataType::Float));
            inputs.push_back(NodeSystem::Pin::createInput("B", NodeSystem::DataType::Float));
            
            outputs.push_back(NodeSystem::Pin::createOutput("Color", NodeSystem::DataType::Vector3));
            
            headerColor = ImVec4(0.3f, 0.6f, 0.3f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int, NodeSystem::EvaluationContext& ctx) override {
            float r = getFloatInput(0, ctx);
            float g = getFloatInput(1, ctx);
            float b = getFloatInput(2, ctx);
            return std::array<float, 3>{r, g, b};
        }
        
        std::string getTypeId() const override { return "Mat.CombineRGB"; }
    };

    // ============================================================================
    // GRAPH CLASS
    // ============================================================================
    
    class MaterialNodeGraph : public NodeSystem::GraphBase {
    public:
        // Helper to find the output node
        MaterialOutputNode* getOutputNode() {
            for (auto& node : nodes) {
                if (auto* out = dynamic_cast<MaterialOutputNode*>(node.get())) {
                    return out; // Return first found
                }
            }
            return nullptr;
        }
        
        // Setup initial default graph
        void createDefaultGraph() {
            clear();
            
            auto* out = addNode<MaterialOutputNode>();
            out->x = 600;
            out->y = 200;
            
            auto* rgb = addNode<RGBNode>();
            rgb->x = 200;
            rgb->y = 200;
            rgb->color = ImVec4(0.05f, 0.6f, 0.8f, 1.0f); // Default Blue-ish
            
            addLink(rgb->outputs[0].id, out->inputs[0].id); // Connect to BaseColor
        }
        
        bool isPinConnected(uint32_t pinId) {
            for(const auto& l : links) {
                if(l.endPinId == pinId) return true;
            }
            return false;
        }
    };

    // ============================================================================
    // BAKING
    // ============================================================================
    
    
    inline BakeResult bakeGraph(MaterialNodeGraph& graph, int w, int h) {
        BakeResult result;
        result.width = w;
        result.height = h;

        MaterialOutputNode* outNode = graph.getOutputNode();
        if (!outNode) return result;

        // Resize buffers based on connection status (optimization)
        if (graph.isPinConnected(outNode->inputs[0].id)) result.albedo.resize(w * h * 4);
        if (graph.isPinConnected(outNode->inputs[1].id)) result.metallic.resize(w * h);
        if (graph.isPinConnected(outNode->inputs[2].id)) result.roughness.resize(w * h);
        if (graph.isPinConnected(outNode->inputs[3].id)) result.emission.resize(w * h * 3);
        if (graph.isPinConnected(outNode->inputs[4].id)) result.normal.resize(w * h * 3);
        
        // Extended PBR
        if (graph.isPinConnected(outNode->inputs[5].id)) result.alpha.resize(w * h);
        if (graph.isPinConnected(outNode->inputs[6].id)) result.transmission.resize(w * h);
        if (graph.isPinConnected(outNode->inputs[7].id)) result.ior.resize(w * h);
        if (graph.isPinConnected(outNode->inputs[8].id)) result.specular.resize(w * h);
        if (graph.isPinConnected(outNode->inputs[9].id)) result.sss_color.resize(w * h * 3);
        if (graph.isPinConnected(outNode->inputs[10].id)) result.sss_scale.resize(w * h);
        if (graph.isPinConnected(outNode->inputs[11].id)) result.clearcoat.resize(w * h);
        if (graph.isPinConnected(outNode->inputs[12].id)) result.clearcoat_roughness.resize(w * h);
        if (graph.isPinConnected(outNode->inputs[13].id)) result.anisotropic.resize(w * h);

        #pragma omp parallel for
        for (int y = 0; y < h; y++) {
            // Thread-local context (stack allocated - no memory leak)
            MaterialContext localMatCtx{w, h, 0, 0};
            NodeSystem::EvaluationContext localCtx(&graph);
            localCtx.setDomainContext(&localMatCtx);
            
            for (int x = 0; x < w; x++) {
                float u = (x + 0.5f) / w;
                float v = (y + 0.5f) / h;
                
                localMatCtx.u = u;
                localMatCtx.v = v;
                
                localCtx.clearCache();
                
                int idx1 = (y * w + x);
                int idx3 = idx1 * 3;
                int idx4 = idx1 * 4;

                // --- ALBEDO ---
                if (!result.albedo.empty()) {
                    std::array<float, 3> c = outNode->getColorInput(0, localCtx);
                    result.albedo[idx4+0] = (unsigned char)(std::clamp(c[0], 0.0f, 1.0f) * 255);
                    result.albedo[idx4+1] = (unsigned char)(std::clamp(c[1], 0.0f, 1.0f) * 255);
                    result.albedo[idx4+2] = (unsigned char)(std::clamp(c[2], 0.0f, 1.0f) * 255);
                    result.albedo[idx4+3] = 255;
                }
                
                // --- METALLIC ---
                if (!result.metallic.empty()) {
                    float f = outNode->getFloatInput(1, localCtx);
                    result.metallic[idx1] = (unsigned char)(std::clamp(f, 0.0f, 1.0f) * 255);
                }
                
                // --- ROUGHNESS ---
                if (!result.roughness.empty()) {
                    float f = outNode->getFloatInput(2, localCtx);
                    result.roughness[idx1] = (unsigned char)(std::clamp(f, 0.0f, 1.0f) * 255);
                }
                
                // --- EMISSION ---
                if (!result.emission.empty()) {
                    std::array<float, 3> c = outNode->getColorInput(3, localCtx);
                    result.emission[idx3+0] = (unsigned char)(std::clamp(c[0], 0.0f, 1.0f) * 255);
                    result.emission[idx3+1] = (unsigned char)(std::clamp(c[1], 0.0f, 1.0f) * 255);
                    result.emission[idx3+2] = (unsigned char)(std::clamp(c[2], 0.0f, 1.0f) * 255);
                }

                // --- NORMAL ---
                if (!result.normal.empty()) {
                     std::array<float, 3> c = outNode->getColorInput(4, localCtx);
                     result.normal[idx3+0] = (unsigned char)(std::clamp(c[0], 0.0f, 1.0f) * 255);
                     result.normal[idx3+1] = (unsigned char)(std::clamp(c[1], 0.0f, 1.0f) * 255);
                     result.normal[idx3+2] = (unsigned char)(std::clamp(c[2], 0.0f, 1.0f) * 255);
                }

                // --- ALPHA ---
                if (!result.alpha.empty()) {
                    float f = outNode->getFloatInput(5, localCtx);
                    result.alpha[idx1] = (unsigned char)(std::clamp(f, 0.0f, 1.0f) * 255);
                }

                // --- TRANSMISSION ---
                if (!result.transmission.empty()) {
                    float f = outNode->getFloatInput(6, localCtx);
                    result.transmission[idx1] = (unsigned char)(std::clamp(f, 0.0f, 1.0f) * 255);
                }

                // --- IOR ---
                if (!result.ior.empty()) {
                    float f = outNode->getFloatInput(7, localCtx);
                    // Map 0-10 IOR range to 0-255? Or just simplified 0-1 transmission weight?
                    // Usually IOR is constant, but if mapped, let's say 1.0-3.0 range mapped to 0-1?
                    // For simply visuals, clamp 0-1 is enough for now, user can adjust range via nodes if needed.
                    // But actually IOR is usually > 1. Let's assume input is raw value, but we store it scaled? 
                    // No, for texture it expects 0-1 usually. Let's clamp 0-10 and div by 10? 
                    // Let's standard: 0.0 - 1.0. 
                    result.ior[idx1] = (unsigned char)(std::clamp(f, 0.0f, 1.0f) * 255);
                }
                
                // --- SPECULAR ---
                if (!result.specular.empty()) {
                    float f = outNode->getFloatInput(8, localCtx);
                    result.specular[idx1] = (unsigned char)(std::clamp(f, 0.0f, 1.0f) * 255);
                }

                // --- SSS COLOR ---
                if (!result.sss_color.empty()) {
                    std::array<float, 3> c = outNode->getColorInput(9, localCtx);
                    result.sss_color[idx3+0] = (unsigned char)(std::clamp(c[0], 0.0f, 1.0f) * 255);
                    result.sss_color[idx3+1] = (unsigned char)(std::clamp(c[1], 0.0f, 1.0f) * 255);
                    result.sss_color[idx3+2] = (unsigned char)(std::clamp(c[2], 0.0f, 1.0f) * 255);
                }

                // --- SSS SCALE ---
                if (!result.sss_scale.empty()) {
                     float f = outNode->getFloatInput(10, localCtx);
                     result.sss_scale[idx1] = (unsigned char)(std::clamp(f, 0.0f, 1.0f) * 255);
                }

                // --- CLEARCOAT ---
                if (!result.clearcoat.empty()) {
                     float f = outNode->getFloatInput(11, localCtx);
                     result.clearcoat[idx1] = (unsigned char)(std::clamp(f, 0.0f, 1.0f) * 255);
                }
                
                // --- CLEARCOAT ROUGHNESS ---
                if (!result.clearcoat_roughness.empty()) {
                     float f = outNode->getFloatInput(12, localCtx);
                     result.clearcoat_roughness[idx1] = (unsigned char)(std::clamp(f, 0.0f, 1.0f) * 255);
                }

                // --- ANISOTROPIC ---
                if (!result.anisotropic.empty()) {
                     float f = outNode->getFloatInput(13, localCtx);
                     result.anisotropic[idx1] = (unsigned char)(std::clamp(f, 0.0f, 1.0f) * 255);
                }
            }
            // No manual delete needed - stack allocated
        }
        return result;
    }
    
} // namespace MaterialNodes

