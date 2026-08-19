/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          Api/RtIpcMethodRegistry.cpp
 * Date:          August 2026
 * License:       MIT
 * =========================================================================
 */

#include "RtIpcMethodRegistry.h"

#include <algorithm>
#include <cctype>
#include <sstream>

MethodRegistry& MethodRegistry::instance() {
    static MethodRegistry s_instance;
    return s_instance;
}

void MethodRegistry::registerMethod(const MethodDescriptor& desc) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_methods.push_back(&desc);
    m_indexed = false;
}

const std::vector<const MethodDescriptor*>& MethodRegistry::all() const {
    ensureIndexed();
    return m_methods;
}

std::vector<const MethodDescriptor*> MethodRegistry::byDomain(const std::string& domain) const {
    ensureIndexed();
    auto it = m_byDomain.find(domain);
    if (it != m_byDomain.end()) {
        return it->second;
    }
    return {};
}

const MethodDescriptor* MethodRegistry::find(const std::string& name) const {
    ensureIndexed();
    auto it = m_byName.find(name);
    if (it != m_byName.end()) {
        return it->second;
    }
    return nullptr;
}

static std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string current;
    for (char c : text) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            current += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        } else if (!current.empty()) {
            tokens.push_back(current);
            current.clear();
        }
    }
    if (!current.empty()) {
        tokens.push_back(current);
    }
    return tokens;
}

std::vector<MethodRegistry::SearchResult> MethodRegistry::search(const std::string& query) const {
    ensureIndexed();
    std::vector<std::string> query_tokens = tokenize(query);
    if (query_tokens.empty()) return {};

    std::vector<SearchResult> results;
    for (const MethodDescriptor* desc : m_methods) {
        int score = 0;
        
        std::string name(desc->name ? desc->name : "");
        std::string summary(desc->summary ? desc->summary : "");
        std::string tags(desc->tags ? desc->tags : "");
        
        std::vector<std::string> target_tokens = tokenize(name + " " + summary + " " + tags);
        
        for (const std::string& qt : query_tokens) {
            for (const std::string& tt : target_tokens) {
                if (tt == qt) {
                    score += 2; // Exact word match
                } else if (tt.find(qt) != std::string::npos || qt.find(tt) != std::string::npos) {
                    // One is a substring of the other (with a minimum length threshold to avoid 'a' matching everything)
                    if (qt.length() > 2 && tt.length() > 2) {
                        score += 1;
                    }
                }
            }
        }
        
        if (score > 0) {
            results.push_back({desc, score});
        }
    }
    
    std::sort(results.begin(), results.end(), [](const SearchResult& a, const SearchResult& b) {
        if (a.score != b.score) return a.score > b.score;
        return std::string(a.desc->name) < std::string(b.desc->name);
    });
    
    return results;
}

std::vector<DomainInfo> MethodRegistry::domains() const {
    ensureIndexed();
    return m_domains;
}

int MethodRegistry::registeredCount() const {
    ensureIndexed();
    return static_cast<int>(m_methods.size());
}

int MethodRegistry::documentedCount() const {
    ensureIndexed();
    int count = 0;
    for (const MethodDescriptor* desc : m_methods)
        if (desc->documented) ++count;
    return count;
}

void MethodRegistry::ensureIndexed() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_indexed) return;
    
    m_byName.clear();
    m_byDomain.clear();
    m_domains.clear();
    
    std::unordered_map<std::string, DomainInfo> domainMap;
    
    for (const MethodDescriptor* desc : m_methods) {
        if (desc->name) m_byName[desc->name] = desc;
        if (desc->domain) {
            m_byDomain[desc->domain].push_back(desc);
            
            auto& dinfo = domainMap[desc->domain];
            if (dinfo.name.empty()) {
                dinfo.name = desc->domain;
                // We'll just use a generic summary if one isn't provided, 
                // but in practice the discovery handler can enrich this.
                dinfo.summary = std::string(desc->domain) + " operations"; 
            }
            dinfo.method_count++;
        }
    }
    
    for (const auto& kv : domainMap) {
        m_domains.push_back(kv.second);
    }
    
    std::sort(m_domains.begin(), m_domains.end(), [](const DomainInfo& a, const DomainInfo& b) {
        return a.name < b.name;
    });
    
    m_indexed = true;
}
