#pragma once

#include "NodeSystem/Node.h"

#include <cstddef>
#include <string>

namespace TerrainNodesV2 {

// Applies the compact, stable-keyed port contract after a node is registered.
// Evaluation remains index-based during the migration; setup/script/IPC can
// move to stable keys independently before obsolete physical pins are removed.
void configureTerrainNodePorts(NodeSystem::NodeBase& node);

NodeSystem::Pin* findTerrainPort(NodeSystem::NodeBase& node,
                                 NodeSystem::PinKind kind,
                                 const char* stableKey);
const NodeSystem::Pin* findTerrainPort(const NodeSystem::NodeBase& node,
                                       NodeSystem::PinKind kind,
                                       const char* stableKey);
uint32_t terrainPortId(const NodeSystem::NodeBase* node,
                       NodeSystem::PinKind kind,
                       const char* stableKey);

const char* pinExposureName(NodeSystem::PinExposure exposure);

// Projects saved before ports carried stable keys can only be resolved by
// slot. That is correct while a contract only GROWS -- most terrain nodes
// appended outputs over time -- and wrong the moment one is PRUNED: Hydraulic
// Erosion published nine outputs and now publishes four, so saved slot 3
// (Discharge) would land on Flow. Both are 0..1 normalised fields of the same
// shape, so nothing would fail; the graph would just be carrying sediment
// transport everywhere it used to carry water discharge.
inline constexpr int kTerrainPortSlotUnknown = -2;  // no legacy table for this type
inline constexpr int kTerrainPortSlotRemoved = -1;  // this port left the contract
int legacyTerrainPortSlot(const std::string& typeId,
                          NodeSystem::PinKind kind,
                          size_t savedSlot);

} // namespace TerrainNodesV2
