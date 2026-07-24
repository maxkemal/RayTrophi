/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          MaterialNodesV2.cpp
* Author:        Kemal Demirtas
* Date:          July 2026
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/

/**
 * @file MaterialNodesV2.cpp
 * @brief NodeRegistry self-registration for the material node graph types.
 *
 * Same pattern as MeshModifiers.cpp (GeoV2.*) / TerrainNodesV2.cpp (TerrainV2.*):
 * one AutoRegisterNode per type id, so deserializeMaterialGraph can re-create
 * nodes by string id and generic "Add Node" menus can list them.
 */

#include "MaterialNodesV2.h"

namespace {
    NodeSystem::AutoRegisterNode<MaterialNodesV2::OutputNode>            reg_MatOutput("MatV2.Output");
    NodeSystem::AutoRegisterNode<MaterialNodesV2::MaterialRefNode>       reg_MatMaterialRef("MatV2.MaterialRef");
    NodeSystem::AutoRegisterNode<MaterialNodesV2::MixMaterialNode>       reg_MatMixMaterial("MatV2.MixMaterial");
    NodeSystem::AutoRegisterNode<MaterialNodesV2::ValueNode>             reg_MatValue("MatV2.Value");
    NodeSystem::AutoRegisterNode<MaterialNodesV2::ColorNode>             reg_MatColor("MatV2.Color");
    NodeSystem::AutoRegisterNode<MaterialNodesV2::TextureCoordinateNode> reg_MatTexCoord("MatV2.TextureCoordinate");
    NodeSystem::AutoRegisterNode<MaterialNodesV2::ImageTextureNode>      reg_MatImageTexture("MatV2.ImageTexture");
    NodeSystem::AutoRegisterNode<MaterialNodesV2::MappingNode>           reg_MatMapping("MatV2.Mapping");
    NodeSystem::AutoRegisterNode<MaterialNodesV2::NoiseTextureNode>      reg_MatNoise("MatV2.Noise");
    NodeSystem::AutoRegisterNode<MaterialNodesV2::GeometryNode>          reg_MatGeometry("MatV2.Geometry");
    NodeSystem::AutoRegisterNode<MaterialNodesV2::ObjectInfoNode>        reg_MatObjectInfo("MatV2.ObjectInfo");
    NodeSystem::AutoRegisterNode<MaterialNodesV2::AttributeNode>         reg_MatAttribute("MatV2.Attribute");
    NodeSystem::AutoRegisterNode<MaterialNodesV2::WaveTextureNode>       reg_MatWave("MatV2.Wave");
    NodeSystem::AutoRegisterNode<MaterialNodesV2::GradientTextureNode>   reg_MatGradient("MatV2.Gradient");
    NodeSystem::AutoRegisterNode<MaterialNodesV2::VectorMathNode>        reg_MatVectorMath("MatV2.VectorMath");
    NodeSystem::AutoRegisterNode<MaterialNodesV2::HueSaturationNode>     reg_MatHueSat("MatV2.HueSaturation");
    NodeSystem::AutoRegisterNode<MaterialNodesV2::RGBCurvesNode>         reg_MatRGBCurves("MatV2.RGBCurves");
    NodeSystem::AutoRegisterNode<MaterialNodesV2::LayerWeightNode>       reg_MatLayerWeight("MatV2.LayerWeight");
    NodeSystem::AutoRegisterNode<MaterialNodesV2::AmbientOcclusionNode>  reg_MatAO("MatV2.AmbientOcclusion");
    NodeSystem::AutoRegisterNode<MaterialNodesV2::BevelNode>             reg_MatBevel("MatV2.Bevel");
    NodeSystem::AutoRegisterNode<MaterialNodesV2::PrincipledVolumeNode>  reg_MatPrincipledVolume("MatV2.PrincipledVolume");
    NodeSystem::AutoRegisterNode<MaterialNodesV2::VolumeInfoNode>        reg_MatVolumeInfo("MatV2.VolumeInfo");
    NodeSystem::AutoRegisterNode<MaterialNodesV2::BlackbodyNode>         reg_MatBlackbody("MatV2.Blackbody");
    NodeSystem::AutoRegisterNode<MaterialNodesV2::VolumeGridNode>        reg_MatVolumeGrid("MatV2.VolumeGrid");
    NodeSystem::AutoRegisterNode<MaterialNodesV2::TimeNode>              reg_MatTime("MatV2.Time");
    NodeSystem::AutoRegisterNode<MaterialNodesV2::CloudShapeNode>        reg_MatCloudShape("MatV2.CloudShape");
    // Legacy typeIds from before Voronoi/Checker were merged into the unified
    // Noise Texture node: create the unified node with the matching kind so
    // old saves keep loading (new saves always write "MatV2.Noise"; old
    // Checker output links are remapped in deserializeMaterialGraph).
    struct RegisterLegacyProceduralPresets {
        RegisterLegacyProceduralPresets() {
            auto& reg = NodeSystem::NodeRegistry::instance();
            reg.registerType("MatV2.Voronoi", [] {
                auto n = std::make_shared<MaterialNodesV2::NoiseTextureNode>();
                n->kind = MaterialNodesV2::NoiseTextureNode::Kind::Voronoi;
                return std::static_pointer_cast<NodeSystem::NodeBase>(n);
            });
            reg.registerType("MatV2.Checker", [] {
                auto n = std::make_shared<MaterialNodesV2::NoiseTextureNode>();
                n->kind = MaterialNodesV2::NoiseTextureNode::Kind::Checker;
                n->scale = 8.0f;  // old Checker's default (its saves omit missing keys)
                return std::static_pointer_cast<NodeSystem::NodeBase>(n);
            });
        }
    } reg_MatLegacyProcedural;
    NodeSystem::AutoRegisterNode<MaterialNodesV2::ColorRampNode>         reg_MatColorRamp("MatV2.ColorRamp");
    NodeSystem::AutoRegisterNode<MaterialNodesV2::MixColorNode>          reg_MatMixColor("MatV2.MixColor");
    NodeSystem::AutoRegisterNode<MaterialNodesV2::InvertNode>            reg_MatInvert("MatV2.Invert");
    NodeSystem::AutoRegisterNode<MaterialNodesV2::GammaNode>             reg_MatGamma("MatV2.Gamma");
    NodeSystem::AutoRegisterNode<MaterialNodesV2::MathNode>              reg_MatMath("MatV2.Math");
    NodeSystem::AutoRegisterNode<MaterialNodesV2::SeparateColorNode>     reg_MatSeparateColor("MatV2.SeparateColor");
    NodeSystem::AutoRegisterNode<MaterialNodesV2::CombineColorNode>      reg_MatCombineColor("MatV2.CombineColor");
    NodeSystem::AutoRegisterNode<MaterialNodesV2::FresnelNode>       reg_MatFresnel("MatV2.Fresnel");
    NodeSystem::AutoRegisterNode<MaterialNodesV2::ClampNode>         reg_MatClamp("MatV2.Clamp");
    NodeSystem::AutoRegisterNode<MaterialNodesV2::FloatCurveNode>        reg_MatFloatCurve("MatV2.FloatCurve");
    NodeSystem::AutoRegisterNode<MaterialNodesV2::MapRangeNode>          reg_MatMapRange("MatV2.MapRange");
    NodeSystem::AutoRegisterNode<MaterialNodesV2::BrightContrastNode>    reg_MatBrightContrast("MatV2.BrightContrast");
    NodeSystem::AutoRegisterNode<MaterialNodesV2::BumpNode>              reg_MatBump("MatV2.Bump");
}
