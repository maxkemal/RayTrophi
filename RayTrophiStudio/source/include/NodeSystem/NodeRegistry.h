/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          NodeRegistry.h
* Author:        Kemal Demirtas
* Date:          July 2026
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once

/**
 * @file NodeRegistry.h
 * @brief Domain-agnostic type-id -> factory registry for NodeBase subclasses.
 *
 * Node.h/Graph.h already let any domain build a typed node graph
 * (NodeSystem::GraphBase::addNode<T>()), but creating a node by its STRING
 * type id (from JSON on load, or from a generic "Add Node" menu that lists
 * every registered type across domains) previously required each domain to
 * hand-maintain its own switch/if-else over a NodeType enum
 * (see TerrainNodeGraphV2::addTerrainNode). That's fine for one domain; it
 * does not scale to "everything is a node" (geometry/scatter/hair/material/
 * simulation node types living in a single master editor), where the editor
 * core cannot know about every domain's enum.
 *
 * NodeRegistry fixes that: any node type, in any domain, self-registers a
 * `typeId -> factory` entry once (at static-init time, via AutoRegisterNode
 * below). No central switch needs to change when a new node type is added
 * anywhere in the codebase.
 */

#include "Node.h"
#include <functional>
#include <unordered_map>
#include <vector>
#include <mutex>

namespace NodeSystem {

    /// Display metadata for one registered node type, used to build "Add Node"
    /// menus without hand-duplicating category/displayName/description at the
    /// registration call site — it's read from the node's own constructor-set
    /// metadata (single source of truth).
    struct NodeTypeInfo {
        std::string typeId;
        std::string category;
        std::string displayName;
        std::string description;
    };

    using NodeFactoryFn = std::function<std::shared_ptr<NodeBase>()>;

    /**
     * @brief Global typeId -> factory registry for NodeBase subclasses.
     *
     * Meyer's singleton (function-local static): safe against static-init-order
     * across translation units — AutoRegisterNode instances in different .cpp
     * files can register in any order, since instance() lazily constructs the
     * registry on first use regardless of which TU calls it first.
     */
    class NodeRegistry {
    public:
        static NodeRegistry& instance() {
            static NodeRegistry reg;
            return reg;
        }

        /**
         * @brief Register a factory for a type id.
         * @returns false if typeId was already registered (first registration
         *          wins — a silent no-op keeps duplicate registration harmless
         *          instead of clobbering an existing factory).
         */
        bool registerType(const std::string& typeId, NodeFactoryFn factory) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (factories_.count(typeId)) return false;
            factories_.emplace(typeId, std::move(factory));
            typeInfoCacheDirty_ = true;
            return true;
        }

        /// Create a new node instance by type id, or nullptr if unregistered.
        std::shared_ptr<NodeBase> create(const std::string& typeId) const {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = factories_.find(typeId);
            if (it == factories_.end()) return nullptr;
            return it->second();
        }

        bool isRegistered(const std::string& typeId) const {
            std::lock_guard<std::mutex> lock(mutex_);
            return factories_.count(typeId) != 0;
        }

        size_t registeredCount() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return factories_.size();
        }

        /**
         * @brief List every registered type's display metadata (for "Add Node"
         * menus). Constructs one throwaway instance per type id to read its
         * metadata — cheap (node constructors are lightweight, this is not a
         * per-frame call) and keeps category/displayName/description sourced
         * from the node's own constructor instead of duplicated strings here.
         */
        std::vector<NodeTypeInfo> getAllTypes() const {
            std::lock_guard<std::mutex> lock(mutex_);
            if (typeInfoCacheDirty_) {
                typeInfoCache_.clear();
                typeInfoCache_.reserve(factories_.size());
                for (const auto& [typeId, factory] : factories_) {
                    NodeTypeInfo info;
                    info.typeId = typeId;
                    if (auto probe = factory()) {
                        info.category = probe->metadata.category;
                        info.displayName = probe->metadata.displayName.empty()
                            ? typeId : probe->metadata.displayName;
                        info.description = probe->metadata.description;
                    } else {
                        info.displayName = typeId;
                    }
                    typeInfoCache_.push_back(std::move(info));
                }
                typeInfoCacheDirty_ = false;
            }
            return typeInfoCache_;
        }

    private:
        NodeRegistry() = default;
        mutable std::mutex mutex_;
        std::unordered_map<std::string, NodeFactoryFn> factories_;
        mutable std::vector<NodeTypeInfo> typeInfoCache_;
        mutable bool typeInfoCacheDirty_ = true;
    };

    /**
     * @brief Zero-boilerplate self-registration for a NodeBase subclass.
     *
     * Usage (file scope, once per node type, in the .cpp that defines it):
     *   static NodeSystem::AutoRegisterNode<HeightmapInputNode>
     *       reg_HeightmapInput("TerrainV2.HeightmapInput");
     *
     * T must be default-constructible (matches the existing addNode<T>()
     * convention already used by every domain's node-creation code).
     */
    template<typename T>
    struct AutoRegisterNode {
        explicit AutoRegisterNode(const std::string& typeId) {
            NodeRegistry::instance().registerType(typeId, [] {
                return std::static_pointer_cast<NodeBase>(std::make_shared<T>());
            });
        }
    };

} // namespace NodeSystem
