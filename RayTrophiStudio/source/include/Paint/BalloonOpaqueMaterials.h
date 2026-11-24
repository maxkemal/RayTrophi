#pragma once

#include "MaterialManager.h"
#include "MaterialProgram.h"
#include "PBRMaterialSnapshot.h"
#include "PrincipledBSDF.h"
#include "Renderer.h"
#include "TriangleMesh.h"
#include "UIContext.h"
#include "json.hpp"
#include "scene_data.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace Paint {

// Balloon meshes own their materials. Keep their authored inputs so switching
// back to the translucent look restores textures and graph outputs as well.
class BalloonOpaqueMaterials {
public:
    void apply(UIContext& context, const nlohmann::json& nodes) {
        using namespace MaterialNodesV2;
        constexpr auto opacity = static_cast<int32_t>(MatSlot::Opacity);
        constexpr auto transmission = static_cast<int32_t>(MatSlot::Transmission);
        constexpr uint32_t glass_slots = (1u << opacity) | (1u << transmission);
        bool programs_changed = false;
        for (const auto& node : nodes) {
            const auto& name = node.get_ref<const std::string&>();
            auto* mesh = context.scene.getFlatNodeMesh(name);
            const auto id = mesh->geometry->get_material_ids()[0];
            auto* material = dynamic_cast<PrincipledBSDF*>(
                MaterialManager::getInstance().getMaterial(id));
            if (!material) {
                throw std::runtime_error("opaque balloon needs a Principled BSDF: " + name);
            }
            const auto& program = material->proceduralProgram;
            const bool suppress_program = program && program->active &&
                (program->drivenSlots & glass_slots) != 0u;
            saved_.push_back({id, material, material->opacityProperty,
                              material->transmissionProperty, material->transmission,
                              material->getTransmissionDensity(), material->getIsBubble(),
                              suppress_program ? program : nullptr});
            material->opacityProperty.alpha = 1.0f;
            material->opacityProperty.texture.reset();
            material->setTransmission(0.0f, material->ior);
            material->transmissionProperty.intensity = 0.0f;
            material->transmissionProperty.texture.reset();
            material->setTransmissionDensity(0.0f);
            material->setIsBubble(false);
            if (suppress_program) {
                auto opaque_program = std::make_shared<MaterialProgram>(*program);
                opaque_program->drivenSlots &= ~glass_slots;
                std::erase_if(opaque_program->instrs, [](const MatInstr& instruction) {
                    return static_cast<MatOp>(instruction.op) == MatOp::Store &&
                        (instruction.aux == opacity || instruction.aux == transmission);
                });
                opaque_program->active = opaque_program->drivenSlots != 0u;
                material->proceduralProgram = std::move(opaque_program);
                programs_changed = true;
            }
            publish(context, id, *material);
        }
        if (programs_changed) {
            context.renderer.updateBackendMaterials(context.scene);
        }
    }

    void restore(UIContext& context) {
        bool programs_changed = false;
        for (const auto& saved : saved_) {
            if (MaterialManager::getInstance().getMaterial(saved.id) != saved.material) {
                continue;
            }
            auto& material = *saved.material;
            material.opacityProperty = saved.opacity;
            material.setTransmission(saved.transmission, material.ior);
            material.transmissionProperty = saved.transmission_property;
            material.setTransmissionDensity(saved.density);
            material.setIsBubble(saved.bubble);
            if (saved.program) {
                material.proceduralProgram = saved.program;
                programs_changed = true;
            }
            publish(context, saved.id, material);
        }
        saved_.clear();
        if (programs_changed) {
            context.renderer.updateBackendMaterials(context.scene);
        }
    }

private:
    static void publish(UIContext& context, uint16_t id, PrincipledBSDF& material) {
        if (material.gpuMaterial) {
            applyPBRMaterialSnapshotToGpuMaterial(capturePBRMaterialSnapshot(material),
                                                  *material.gpuMaterial);
        }
        context.renderer.updateBackendMaterial(context.scene, id);
    }

    struct Saved {
        uint16_t id;
        PrincipledBSDF* material;
        MaterialProperty opacity;
        MaterialProperty transmission_property;
        float transmission;
        float density;
        bool bubble;
        std::shared_ptr<MaterialNodesV2::MaterialProgram> program;
    };
    std::vector<Saved> saved_;
};

} // namespace Paint
