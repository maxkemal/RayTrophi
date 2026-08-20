/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          Api/RtIpcMethodRegistry.h
 * Date:          August 2026
 * License:       MIT
 * =========================================================================
 * Static method registry for agent-oriented API discovery.
 *
 * Every IPC method declares a MethodDescriptor at its dispatch site via a
 * static MethodRegistration object.  The global MethodRegistry singleton
 * collects these descriptors at static-init time so the agent.* discovery
 * endpoints can answer "what methods exist?" without a hand-maintained
 * catalogue that drifts out of sync.
 *
 * Runtime cost: zero after startup (all data is const char*; no heap alloc).
 * =========================================================================
 */

#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_map>

// ---------------------------------------------------------------------------
// MethodParam — one parameter of an IPC method
// ---------------------------------------------------------------------------
struct MethodParam {
    const char* name;           // "domain_min"
    const char* type;           // "string"|"float"|"int"|"bool"|"vec3"|"matrix"|"object"|"array"
    bool        required;
    const char* description;    // short human/agent-readable text
    const char* default_value;  // nullptr if no default; otherwise JSON-like string
    const char* enum_values;    // nullptr or "fluid|gas"
};

// ---------------------------------------------------------------------------
// MethodDescriptor — full metadata for one IPC method
// ---------------------------------------------------------------------------
struct MethodDescriptor {
    const char* name;           // "fluid.create"
    const char* domain;         // "fluid"
    const char* summary;        // one-line description
    const char* notes;          // nullable; multi-line detail
    const char* access;         // "read"|"write"|"render"|"admin"
    const char* capability;     // exact security capability, e.g. "SceneWrite"
    bool        undoable;
    const char* return_type;    // "FluidDomainInfo" | "bool" | "string[]" | "void"
    const char* tags;           // pipe-separated: "simulation|fluid|create|domain"
    const char* related;        // pipe-separated: "fluid.get|fluid.update"

    // ★★ Sequencing. `related` answers "what else is nearby"; these four answer
    // "what ORDER". A caller with no prior knowledge of this application — the
    // case a small local model is always in — cannot infer that a domain must
    // exist before it can be configured, or that a solver change means the bake
    // is no longer valid. Left implicit, that knowledge lived only in the head
    // of the person who wrote the dispatch.
    // All four are pipe-separated method names, or nullptr.
    const char* prerequisites;  // must succeed BEFORE this call ("fluid.create_domain")
    const char* next_steps;     // what usually follows ("gas.set_shader|timeline.set_frame")
    const char* verify_with;    // how to CHECK it landed ("gas.get_settings|render.probe")
    // ★ Named state this call silently makes stale. The most expensive class of
    // bug in this repo is a stale artefact that still answers plausibly, so the
    // invalidation has to be part of the method's description, not folklore.
    const char* invalidates;    // "simulation_cache|tlas" etc., or nullptr

    const MethodParam* params;
    int         param_count;
    // False when the method carries an extracted parameter schema but no
    // hand-written summary. agent.discover reports the share of `true` as
    // documented_coverage, so a gap is measured instead of hidden.
    bool        documented;
};

// ---------------------------------------------------------------------------
// DomainInfo — aggregated per-domain summary (computed on first access)
// ---------------------------------------------------------------------------
struct DomainInfo {
    std::string name;
    std::string summary;
    int method_count = 0;
};

// ---------------------------------------------------------------------------
// MethodRegistry — global singleton; populated at static-init time
// ---------------------------------------------------------------------------
class MethodRegistry {
public:
    static MethodRegistry& instance();

    // Registration (called from static initializers)
    void registerMethod(const MethodDescriptor& desc);

    // Queries
    const std::vector<const MethodDescriptor*>& all() const;
    std::vector<const MethodDescriptor*> byDomain(const std::string& domain) const;
    const MethodDescriptor* find(const std::string& name) const;

    // Keyword search: tokenize query, match against tags + summaries
    struct SearchResult {
        const MethodDescriptor* desc;
        int score;
    };
    std::vector<SearchResult> search(const std::string& query) const;

    // Domain listing
    std::vector<DomainInfo> domains() const;

    // Coverage metrics. registeredCount() equals the number of dispatched
    // methods by construction: the descriptor table is generated from the
    // dispatch sources and scripts/audit_ipc_capabilities.py fails when the two
    // drift. documentedCount() is the one that can legitimately be lower.
    int registeredCount() const;
    int documentedCount() const;

private:
    MethodRegistry() = default;
    void ensureIndexed() const;

    mutable std::mutex              m_mutex;
    std::vector<const MethodDescriptor*> m_methods;

    // Lazily built indices
    mutable bool                    m_indexed = false;
    mutable std::unordered_map<std::string, const MethodDescriptor*>  m_byName;
    mutable std::unordered_map<std::string, std::vector<const MethodDescriptor*>> m_byDomain;
    mutable std::vector<DomainInfo> m_domains;
};

// ---------------------------------------------------------------------------
// MethodRegistration — RAII helper for static-init registration
// ---------------------------------------------------------------------------
struct MethodRegistration {
    explicit MethodRegistration(const MethodDescriptor& desc) {
        MethodRegistry::instance().registerMethod(desc);
    }
};
