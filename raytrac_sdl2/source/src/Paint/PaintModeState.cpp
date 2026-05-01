#include "Paint/PaintModeState.h"
#include <algorithm>

namespace Paint {

PaintModeState::PaintModeState() {
    ui_layers.push_back(PaintLayer{});
    linked_channels.fill(false);
}

void PaintModeState::setAdapter(const PaintSurfaceAdapterPtr& adapter) {
    if (!active_target_name.empty() && active_layer_id != 0) {
        last_layer_id_by_target[active_target_name] = active_layer_id;
    }

    active_adapter_ = adapter;
    bound_layer_stack_ = nullptr;  // Unbind old object's layer stack
    active_target_name = (active_adapter_ && active_adapter_->isValid())
        ? active_adapter_->getTarget().display_name
        : std::string();
    auto remembered = last_layer_id_by_target.find(active_target_name);
    active_layer_id = (remembered != last_layer_id_by_target.end()) ? remembered->second : 0;
    syncLayersFromAdapter();
}

void PaintModeState::clearAdapter() {
    if (!active_target_name.empty() && active_layer_id != 0) {
        last_layer_id_by_target[active_target_name] = active_layer_id;
    }

    active_adapter_.reset();
    active_target_name.clear();
    bound_layer_stack_ = nullptr;
    ui_layers.clear();
    ui_layers.push_back(PaintLayer{});
    active_layer_index = 0;
    active_layer_id = 0;
    linked_channels.fill(false);
    stroke = PaintStroke{};
}

PaintSurfaceAdapterPtr PaintModeState::getAdapter() const {
    return active_adapter_;
}

bool PaintModeState::hasValidTarget() const {
    return active_adapter_ && active_adapter_->isValid();
}

void PaintModeState::syncLayersFromAdapter() {
    ui_layers.clear();

    if (!active_adapter_ || !active_adapter_->isValid()) {
        ui_layers.push_back(PaintLayer{});
        active_layer_index = 0;
        active_layer_id = 0;
        return;
    }

    const int layer_count = std::max(0, active_adapter_->getLayerCount());
    for (int i = 0; i < layer_count; ++i) {
        PaintLayer layer;
        layer.name = active_adapter_->getLayerName(i);
        ui_layers.push_back(layer);
    }

    if (ui_layers.empty()) {
        ui_layers.push_back(PaintLayer{});
    }

    clampActiveLayer();
}

void PaintModeState::clampActiveLayer() {
    if (ui_layers.empty()) {
        active_layer_index = 0;
        return;
    }

    active_layer_index = std::clamp(active_layer_index, 0, static_cast<int>(ui_layers.size()) - 1);
}

// ======================== Layer Stack bridge ========================

void PaintModeState::bindLayerStack(PaintLayerStack* stack) {
    bound_layer_stack_ = stack;
    if (stack) {
        syncLayersFromStack();
    }
}

void PaintModeState::syncLayersFromStack() {
    ui_layers.clear();

    if (!bound_layer_stack_ || bound_layer_stack_->empty()) {
        ui_layers.push_back(PaintLayer{});
        active_layer_index = 0;
        active_layer_id = 0;
        return;
    }

    for (int i = 0; i < bound_layer_stack_->layerCount(); ++i) {
        const PaintLayerData* ld = bound_layer_stack_->layerAt(i);
        if (ld) {
            ui_layers.push_back(ld->meta);
        }
    }

    if (ui_layers.empty()) {
        ui_layers.push_back(PaintLayer{});
    }

    int restored_index = -1;
    if (active_layer_id != 0) {
        restored_index = bound_layer_stack_->indexOfId(active_layer_id);
    }
    if (restored_index >= 0) {
        active_layer_index = restored_index;
    } else {
        clampActiveLayer();
    }

    if (const PaintLayerData* active = bound_layer_stack_->layerAt(active_layer_index)) {
        active_layer_id = active->id;
        if (!active_target_name.empty()) {
            last_layer_id_by_target[active_target_name] = active_layer_id;
        }
    } else {
        active_layer_id = 0;
    }
}

void PaintModeState::pushLayerMetaToStack() {
    if (!bound_layer_stack_) return;

    for (int i = 0; i < static_cast<int>(ui_layers.size()) && i < bound_layer_stack_->layerCount(); ++i) {
        PaintLayerData* ld = bound_layer_stack_->layerAt(i);
        if (ld) {
            ld->meta = ui_layers[i];
        }
    }
}

int PaintModeState::addLayerAboveCurrent(const std::string& name) {
    if (!bound_layer_stack_) return -1;
    const int insert_at = active_layer_index + 1;
    const int idx = bound_layer_stack_->addLayer(name, insert_at);
    syncLayersFromStack();
    active_layer_index = idx;
    clampActiveLayer();
    if (const PaintLayerData* active = bound_layer_stack_->layerAt(active_layer_index)) {
        active_layer_id = active->id;
        if (!active_target_name.empty()) {
            last_layer_id_by_target[active_target_name] = active_layer_id;
        }
    }
    return idx;
}

int PaintModeState::duplicateCurrentLayer() {
    if (!bound_layer_stack_) return -1;
    const int idx = bound_layer_stack_->duplicateLayer(active_layer_index);
    if (idx >= 0) {
        syncLayersFromStack();
        active_layer_index = idx;
        clampActiveLayer();
        if (const PaintLayerData* active = bound_layer_stack_->layerAt(active_layer_index)) {
            active_layer_id = active->id;
            if (!active_target_name.empty()) {
                last_layer_id_by_target[active_target_name] = active_layer_id;
            }
        }
    }
    return idx;
}

bool PaintModeState::removeCurrentLayer() {
    if (!bound_layer_stack_) return false;
    const int previous_index = active_layer_index;
    if (!bound_layer_stack_->removeLayer(active_layer_index)) return false;
    active_layer_id = 0;
    active_layer_index = std::min(previous_index, bound_layer_stack_->layerCount() - 1);
    syncLayersFromStack();
    clampActiveLayer();
    if (const PaintLayerData* active = bound_layer_stack_->layerAt(active_layer_index)) {
        active_layer_id = active->id;
        if (!active_target_name.empty()) {
            last_layer_id_by_target[active_target_name] = active_layer_id;
        }
    }
    return true;
}

bool PaintModeState::mergeCurrentLayerDown() {
    if (!bound_layer_stack_) return false;
    if (!bound_layer_stack_->mergeDown(active_layer_index)) return false;
    active_layer_index = std::max(0, active_layer_index - 1);
    active_layer_id = 0;
    syncLayersFromStack();
    clampActiveLayer();
    if (const PaintLayerData* active = bound_layer_stack_->layerAt(active_layer_index)) {
        active_layer_id = active->id;
        if (!active_target_name.empty()) {
            last_layer_id_by_target[active_target_name] = active_layer_id;
        }
    }
    return true;
}

bool PaintModeState::moveCurrentLayer(int delta) {
    if (!bound_layer_stack_) return false;
    const int target = active_layer_index + delta;
    if (!bound_layer_stack_->moveLayer(active_layer_index, target)) return false;
    active_layer_index = target;
    if (const PaintLayerData* active = bound_layer_stack_->layerAt(active_layer_index)) {
        active_layer_id = active->id;
        if (!active_target_name.empty()) {
            last_layer_id_by_target[active_target_name] = active_layer_id;
        }
    }
    syncLayersFromStack();
    clampActiveLayer();
    return true;
}

void PaintModeState::flattenAllLayers() {
    if (!bound_layer_stack_) return;
    bound_layer_stack_->flattenAll();
    active_layer_index = 0;
    active_layer_id = 0;
    syncLayersFromStack();
}

} // namespace Paint
