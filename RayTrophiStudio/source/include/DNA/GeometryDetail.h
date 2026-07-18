#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <stdexcept>
#include <mutex>
#include <type_traits>
#include <array>
#include <cstdint>
#include <algorithm>
#include "AlignedAllocator.h"
#include "Vec3.h"
#include "Vec2.h"

namespace DNA {

    /**
     * @brief Enum-indexed identity for the fixed set of core geometry channels.
     *
     * Core channels live in O(1) array slots (no hash, no string compare) and keep a
     * STABLE slot order for zero-copy GPU upload. Custom/named attributes (paint layers,
     * vertex groups, AI-assigned semantic channels) keep the flexible string map.
     * The string-keyed public API (add_attribute/get_attribute_data/...) is preserved as a
     * thin compatibility shim that resolves core names to slots via core_attr_index().
     */
    enum class Attr : uint8_t {
        P = 0,        // Vec3   current/world position
        N,            // Vec3   current/world normal
        UV,           // Vec2   uv set 0
        P_orig,       // Vec3   bind/original position
        N_orig,       // Vec3   bind/original normal
        MaterialID,   // uint16 per-vertex material id
        CoreCount
    };

    // Resolve a string attribute name to its core slot index, or -1 if it is a custom attribute.
    inline int core_attr_index(const std::string& name) noexcept {
        switch (name.size()) {
            case 1:
                if (name[0] == 'P') return static_cast<int>(Attr::P);
                if (name[0] == 'N') return static_cast<int>(Attr::N);
                return -1;
            case 2:
                if (name[0] == 'u' && name[1] == 'v') return static_cast<int>(Attr::UV);
                return -1;
            case 6:
                if (name == "P_orig") return static_cast<int>(Attr::P_orig);
                if (name == "N_orig") return static_cast<int>(Attr::N_orig);
                return -1;
            case 10:
                if (name == "materialID") return static_cast<int>(Attr::MaterialID);
                return -1;
            default:
                return -1;
        }
    }

    /**
     * @brief Abstract base for attribute buffers.
     * Allows polymorphic handling of raw flat arrays in the engine core.
     */
    class AttributeBuffer {
    public:
        virtual ~AttributeBuffer() = default;
        virtual void* raw_data() = 0;
        virtual const void* raw_data() const = 0;
        virtual size_t size_in_bytes() const = 0;
        virtual size_t element_count() const = 0;
        virtual void resize(size_t count) = 0;
        virtual std::unique_ptr<AttributeBuffer> clone() const = 0;
    };

    /**
     * @brief 32-byte aligned attribute buffer.
     * Enforces AVX256 alignment for high-performance vector math.
     */
    template <typename T>
    class AlignedAttributeBuffer final : public AttributeBuffer {
    public:
        std::vector<T, AlignedAllocator<T, 32>> data;

        void* raw_data() override { return data.data(); }
        const void* raw_data() const override { return data.data(); }
        size_t size_in_bytes() const override { return data.size() * sizeof(T); }
        size_t element_count() const override { return data.size(); }
        void resize(size_t count) override { data.resize(count); }
        
        std::unique_ptr<AttributeBuffer> clone() const override {
            auto copy = std::make_unique<AlignedAttributeBuffer<T>>();
            copy->data = this->data;
            return copy;
        }
    };

    /**
     * @brief Polymorphic representation of a single attribute change (Delta).
     * Enables immutable geometry snapshots with zero-overhead Undo/Redo and Replays.
     */
    class AttributeDelta {
    public:
        virtual ~AttributeDelta() = default;
        virtual const std::string& attribute_name() const = 0;
        virtual void apply(AttributeBuffer* base_buffer) const = 0;
        virtual std::unique_ptr<AttributeDelta> clone() const = 0;
    };

    /**
     * @brief Strongly-typed attribute delta.
     * Stores sparse changes (indices and new values/offsets).
     */
    template <typename T>
    class TypedAttributeDelta final : public AttributeDelta {
    private:
        std::string name;
        std::vector<uint32_t> indices;
        std::vector<T> values;
        bool is_relative_offset; // true => add to base, false => overwrite base

    public:
        TypedAttributeDelta(const std::string& attr_name, 
                            std::vector<uint32_t>&& idx, 
                            std::vector<T>&& val, 
                            bool relative = false)
            : name(attr_name), indices(std::move(idx)), values(std::move(val)), is_relative_offset(relative) {}

        const std::string& attribute_name() const override { return name; }

        void apply(AttributeBuffer* base_buffer) const override {
            auto typed_buf = dynamic_cast<AlignedAttributeBuffer<T>*>(base_buffer);
            if (!typed_buf) return;

            auto& data = typed_buf->data;
            size_t n = indices.size();
            
            if (is_relative_offset) {
                // AVX compiler-friendly addition loop
                for (size_t i = 0; i < n; ++i) {
                    uint32_t idx = indices[i];
                    if (idx < data.size()) {
                        data[idx] += values[i];
                    }
                }
            } else {
                // Overwrite loop
                for (size_t i = 0; i < n; ++i) {
                    uint32_t idx = indices[i];
                    if (idx < data.size()) {
                        data[idx] = values[i];
                    }
                }
            }
        }

        std::unique_ptr<AttributeDelta> clone() const override {
            return std::make_unique<TypedAttributeDelta<T>>(name, std::vector<uint32_t>(indices), std::vector<T>(values), is_relative_offset);
        }
    };

    /**
     * @brief GeometryDetail represents the core, flat, SoA geometry database.
     * 
     * Supports:
     * - Dynamic attribute allocation (positions "P", normals "N", UVs "uv", paint layers).
     * - Immutable base snapshot + Delta stack tracking.
     * - Zero-copy raw data streaming for Vulkan RT and Embree.
     */
    class GeometryDetail {
    private:
        static constexpr size_t kCoreCount = static_cast<size_t>(Attr::CoreCount);
        using CoreSlots = std::array<std::unique_ptr<AttributeBuffer>, kCoreCount>;

        // Core channels in fixed enum-indexed slots (O(1), no hash). Base = undeformed
        // snapshot; active = lazily-evaluated clone, ONLY populated when deltas exist.
        CoreSlots core_base;
        mutable CoreSlots core_active;

        // Custom/named attributes (paint layers, groups, semantic channels) keep the map.
        std::unordered_map<std::string, std::unique_ptr<AttributeBuffer>> custom_base;
        mutable std::unordered_map<std::string, std::unique_ptr<AttributeBuffer>> custom_active;

        mutable bool active_state_dirty = false;
        mutable std::mutex active_state_mutex;

        // Delta history stack (enables instant Undo/Redo and Replay)
        std::vector<std::unique_ptr<AttributeDelta>> delta_stack;

        size_t vertex_count = 0;

        // Cached raw pointers for zero-overhead hot path access
        mutable const Vec3* cached_P = nullptr;
        mutable const Vec3* cached_N = nullptr;
        mutable const Vec2* cached_uv = nullptr;
        mutable const Vec3* cached_P_orig = nullptr;
        mutable const Vec3* cached_N_orig = nullptr;
        mutable const uint16_t* cached_materialID = nullptr;
        mutable bool cached_pointers_dirty = true;

        template <typename T>
        static const T* raw_from_slot(const CoreSlots& slots, Attr a) noexcept {
            const auto& buf = slots[static_cast<size_t>(a)];
            return buf ? reinterpret_cast<const T*>(buf->raw_data()) : nullptr;
        }

        void evaluate_active_state_internal() const {
            for (size_t i = 0; i < kCoreCount; ++i) {
                core_active[i] = core_base[i] ? core_base[i]->clone() : nullptr;
            }
            custom_active.clear();
            for (const auto& [name, buf] : custom_base) {
                if (buf) custom_active[name] = buf->clone();
            }
            for (const auto& delta : delta_stack) {
                if (!delta) continue;
                const std::string& nm = delta->attribute_name();
                int ci = core_attr_index(nm);
                if (ci >= 0) {
                    if (core_active[static_cast<size_t>(ci)]) {
                        delta->apply(core_active[static_cast<size_t>(ci)].get());
                    }
                } else {
                    auto it = custom_active.find(nm);
                    if (it != custom_active.end() && it->second) {
                        delta->apply(it->second.get());
                    }
                }
            }
            active_state_dirty = false;
        }

        void update_cached_pointers_internal() const {
            const CoreSlots& core = delta_stack.empty() ? core_base : core_active;
            cached_P          = raw_from_slot<Vec3>(core, Attr::P);
            cached_N          = raw_from_slot<Vec3>(core, Attr::N);
            cached_uv         = raw_from_slot<Vec2>(core, Attr::UV);
            cached_P_orig     = raw_from_slot<Vec3>(core, Attr::P_orig);
            cached_N_orig     = raw_from_slot<Vec3>(core, Attr::N_orig);
            cached_materialID = raw_from_slot<uint16_t>(core, Attr::MaterialID);
            cached_pointers_dirty = false;
        }

        // Coherent re-evaluation gate. cached_pointers_dirty is set ONLY when the buffer
        // composition changes (add_attribute / resize / push|pop|clear delta), NEVER on an
        // in-place value write — so a normal sculpt/skinning/transform write does not trigger
        // a re-clone here. When no deltas exist, base IS the active state: we skip the clone
        // entirely (update_cached_pointers_internal points cached_* straight at core_base).
        void evaluate_active_state() const {
            if (!cached_pointers_dirty) return;
            std::lock_guard<std::mutex> lock(active_state_mutex);
            if (!cached_pointers_dirty) return;
            if (!delta_stack.empty()) {
                evaluate_active_state_internal();
            } else {
                for (auto& p : core_active) p.reset();
                custom_active.clear();
                active_state_dirty = false;
            }
            update_cached_pointers_internal();
        }

    public:
        // High-performance direct accessors bypassing map lookups and string comparisons
        inline const Vec3* get_positions() const noexcept {
            if (cached_pointers_dirty) evaluate_active_state();
            return cached_P;
        }
        inline const Vec3* get_normals() const noexcept {
            if (cached_pointers_dirty) evaluate_active_state();
            return cached_N;
        }
        inline const Vec2* get_uvs() const noexcept {
            if (cached_pointers_dirty) evaluate_active_state();
            return cached_uv;
        }
        inline const Vec3* get_positions_orig() const noexcept {
            if (cached_pointers_dirty) evaluate_active_state();
            return cached_P_orig;
        }
        inline const Vec3* get_normals_orig() const noexcept {
            if (cached_pointers_dirty) evaluate_active_state();
            return cached_N_orig;
        }
        inline const uint16_t* get_material_ids() const noexcept {
            if (cached_pointers_dirty) evaluate_active_state();
            return cached_materialID;
        }

        // Writable accessors return a pointer into the (already-evaluated) active buffer and
        // perform an IN-PLACE write. They must NOT dirty the cache: the buffer is not moved,
        // so cached_P/N/uv stay valid. (Dirtying here was the re-clone storm — fixed in Faz 0.)
        inline Vec3* get_positions_mut() noexcept {
            if (cached_pointers_dirty) evaluate_active_state();
            return const_cast<Vec3*>(cached_P);
        }

        inline Vec3* get_normals_mut() noexcept {
            if (cached_pointers_dirty) evaluate_active_state();
            return const_cast<Vec3*>(cached_N);
        }

        inline Vec2* get_uvs_mut() noexcept {
            if (cached_pointers_dirty) evaluate_active_state();
            return const_cast<Vec2*>(cached_uv);
        }

        // Flat Index Buffer (AVX & GPU friendly)
        std::vector<uint32_t, AlignedAllocator<uint32_t, 32>> indices;

        // Bone indices and weights for skinning (flat, per-vertex)
        std::vector<std::vector<std::pair<int, float>>> skin_weights;
        
        // Bone hash to avoid redundant skinning calculations
        uint64_t last_skinned_pose_hash = 0;

        GeometryDetail() = default;

        // Support deep-copying of the entire geometry database (safe cloning)
        GeometryDetail(const GeometryDetail& other) {
            vertex_count = other.vertex_count;
            indices = other.indices;
            skin_weights = other.skin_weights;
            last_skinned_pose_hash = other.last_skinned_pose_hash;
            active_state_dirty = true;
            cached_pointers_dirty = true;

            for (size_t i = 0; i < kCoreCount; ++i) {
                core_base[i] = other.core_base[i] ? other.core_base[i]->clone() : nullptr;
            }
            for (const auto& [name, buf] : other.custom_base) {
                if (buf) custom_base[name] = buf->clone();
            }
            for (const auto& delta : other.delta_stack) {
                delta_stack.push_back(delta->clone());
            }
        }

        GeometryDetail& operator=(const GeometryDetail& other) {
            if (this == &other) return *this;

            vertex_count = other.vertex_count;
            indices = other.indices;
            skin_weights = other.skin_weights;
            last_skinned_pose_hash = other.last_skinned_pose_hash;
            for (auto& p : core_base) p.reset();
            for (auto& p : core_active) p.reset();
            custom_base.clear();
            custom_active.clear();
            delta_stack.clear();
            active_state_dirty = true;
            cached_pointers_dirty = true;

            for (size_t i = 0; i < kCoreCount; ++i) {
                core_base[i] = other.core_base[i] ? other.core_base[i]->clone() : nullptr;
            }
            for (const auto& [name, buf] : other.custom_base) {
                if (buf) custom_base[name] = buf->clone();
            }
            for (const auto& delta : other.delta_stack) {
                delta_stack.push_back(delta->clone());
            }
            return *this;
        }

        /**
         * @brief Allocates a new attribute array in the base snapshot.
         */
        template <typename T>
        void add_attribute(const std::string& name) {
            auto buf = std::make_unique<AlignedAttributeBuffer<T>>();
            buf->resize(vertex_count);
            int ci = core_attr_index(name);
            if (ci >= 0) core_base[static_cast<size_t>(ci)] = std::move(buf);
            else         custom_base[name] = std::move(buf);
            active_state_dirty = true;
            cached_pointers_dirty = true;
        }

        inline bool has_attribute(const std::string& name) const noexcept {
            int ci = core_attr_index(name);
            if (ci >= 0) return core_base[static_cast<size_t>(ci)] != nullptr;
            return custom_base.find(name) != custom_base.end();
        }

        // Removes only a named/custom attribute; canonical P/N/uv/etc. cannot
        // be removed through this API. Used by derived field publishers when an
        // output pin is disconnected so foliage never samples stale data.
        bool remove_custom_attribute(const std::string& name) {
            if (core_attr_index(name) >= 0) return false;
            const bool removed = custom_base.erase(name) > 0;
            custom_active.erase(name);
            if (removed) {
                active_state_dirty = true;
                cached_pointers_dirty = true;
            }
            return removed;
        }

        /**
         * @brief Names of all custom/named attributes (paint layers, masks, groups) —
         * add_attribute() always writes custom_base directly regardless of delta_stack
         * state, so this is authoritative even mid-delta. Used to populate attribute
         * picker dropdowns (UI) instead of free-text entry.
         */
        std::vector<std::string> listCustomAttributeNames() const {
            std::vector<std::string> names;
            names.reserve(custom_base.size());
            for (const auto& [name, buf] : custom_base) {
                if (buf) names.push_back(name);
            }
            std::sort(names.begin(), names.end());
            return names;
        }

        /**
         * @brief Returns a pointer to the active/current attribute data.
         * If no deltas are present, returns the base snapshot directly (zero overhead!).
         */
        template <typename T>
        const T* get_attribute_data(const std::string& name) const {
            if (cached_pointers_dirty) evaluate_active_state();

            // Core fast path: enum slot → cached pointer (no hash, no string compare).
            int ci = core_attr_index(name);
            if (ci >= 0) {
                switch (static_cast<Attr>(ci)) {
                    case Attr::P:          return reinterpret_cast<const T*>(cached_P);
                    case Attr::N:          return reinterpret_cast<const T*>(cached_N);
                    case Attr::UV:         return reinterpret_cast<const T*>(cached_uv);
                    case Attr::P_orig:     return reinterpret_cast<const T*>(cached_P_orig);
                    case Attr::N_orig:     return reinterpret_cast<const T*>(cached_N_orig);
                    case Attr::MaterialID: return reinterpret_cast<const T*>(cached_materialID);
                    default:               break;
                }
            }

            // Custom attribute lookup
            const auto& map = delta_stack.empty() ? custom_base : custom_active;
            auto it = map.find(name);
            if (it == map.end() || !it->second) return nullptr;
            return reinterpret_cast<const T*>(it->second->raw_data());
        }

        // In-place writable access. Like get_positions_mut: does NOT dirty the cache (the
        // buffer is not reallocated). Returns base buffer when no deltas, else the evaluated
        // active clone.
        template <typename T>
        T* get_attribute_data_mut(const std::string& name) {
            // Core fast path: cached_* already point at the right (base or active) slot.
            int ci = core_attr_index(name);
            if (ci >= 0) {
                if (cached_pointers_dirty) evaluate_active_state();
                switch (static_cast<Attr>(ci)) {
                    case Attr::P:          return const_cast<T*>(reinterpret_cast<const T*>(cached_P));
                    case Attr::N:          return const_cast<T*>(reinterpret_cast<const T*>(cached_N));
                    case Attr::UV:         return const_cast<T*>(reinterpret_cast<const T*>(cached_uv));
                    case Attr::P_orig:     return const_cast<T*>(reinterpret_cast<const T*>(cached_P_orig));
                    case Attr::N_orig:     return const_cast<T*>(reinterpret_cast<const T*>(cached_N_orig));
                    case Attr::MaterialID: return const_cast<T*>(reinterpret_cast<const T*>(cached_materialID));
                    default:               break;
                }
            }

            // Custom attribute: in-place write, does NOT dirty the cache (buffer not moved).
            if (delta_stack.empty()) {
                auto it = custom_base.find(name);
                if (it == custom_base.end()) return nullptr;
                return reinterpret_cast<T*>(it->second->raw_data());
            }
            if (cached_pointers_dirty) evaluate_active_state();
            auto it = custom_active.find(name);
            if (it == custom_active.end()) return nullptr;
            return reinterpret_cast<T*>(it->second->raw_data());
        }

        /**
         * @brief Appends a new delta to the modification stack.
         */
        template <typename T>
        void push_delta(const std::string& attr_name, 
                        std::vector<uint32_t>&& idx, 
                        std::vector<T>&& val, 
                        bool relative = false) 
        {
            delta_stack.push_back(std::make_unique<TypedAttributeDelta<T>>(
                attr_name, std::move(idx), std::move(val), relative
            ));
            active_state_dirty = true;
            cached_pointers_dirty = true;
        }

        /**
         * @brief Pops the last delta from the stack (Instant O(1) Undo!).
         */
        void pop_delta() {
            if (!delta_stack.empty()) {
                delta_stack.pop_back();
                active_state_dirty = true;
                cached_pointers_dirty = true;
            }
        }

        inline size_t delta_count() const noexcept {
            return delta_stack.size();
        }

        void clear_deltas() noexcept {
            delta_stack.clear();
            for (auto& p : core_active) p.reset();
            custom_active.clear();
            active_state_dirty = false;
            cached_pointers_dirty = true;
        }

        /**
         * @brief Bake the current ACTIVE state (base + all deltas) into the base
         * snapshot and clear the delta stack. Visually a no-op — the data the
         * accessors return is unchanged — but afterwards in-place writes land in
         * base and therefore SURVIVE copy-construction. Needed by consumers that
         * deep-copy a mesh and then mutate it (Geo-DAG nodes): with a non-empty
         * delta stack, their writes would go into the active clone, which the NEXT
         * copy-construction silently discards (it copies base + deltas only).
         */
        void flatten_deltas() {
            if (delta_stack.empty()) return;
            {
                std::lock_guard<std::mutex> lock(active_state_mutex);
                evaluate_active_state_internal();
                for (size_t i = 0; i < kCoreCount; ++i) {
                    if (core_active[i]) core_base[i] = std::move(core_active[i]);
                }
                for (auto& [name, buf] : custom_active) {
                    if (buf) custom_base[name] = std::move(buf);
                }
                custom_active.clear();
                delta_stack.clear();
                active_state_dirty = false;
                cached_pointers_dirty = true;
            }
        }

        /**
         * @brief Resizes all base attribute arrays (core slots + custom map).
         */
        void resize_vertices(size_t new_count) {
            vertex_count = new_count;
            for (auto& buf : core_base) {
                if (buf) buf->resize(new_count);
            }
            for (auto& [name, buf] : custom_base) {
                if (buf) buf->resize(new_count);
            }
            active_state_dirty = true;
            cached_pointers_dirty = true;
        }

        inline size_t get_vertex_count() const noexcept {
            return vertex_count;
        }
    };

} // namespace DNA
