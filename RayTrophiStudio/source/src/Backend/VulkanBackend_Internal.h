#pragma once
// ============================================================================
// VulkanBackend_Internal.h
//
// Implementation-private helpers shared between the VulkanBackend translation
// units (VulkanBackend.cpp, VulkanBackend_Raster.cpp, ...). This header is NOT
// part of the public backend interface: nothing outside src/Backend/ may
// include it, and nothing here belongs in Backend/VulkanBackend.h.
//
// These used to live in an anonymous namespace at the top of the (20k line)
// VulkanBackend.cpp. Anonymous-namespace internal linkage is exactly what makes
// that file hard to split, so helpers needed by more than one unit are promoted
// to `inline` here instead of being duplicated.
// ============================================================================

#include <string>

namespace VulkanRT {
namespace detail {

// An instance belongs to `queryNodeName` when its own node name matches
// outright, or when it is one of the per-material splits the importer emits as
// "<node>_mat_<n>".
inline bool matchesNodeNameForInstance(const std::string& instanceNodeName,
                                       const std::string& queryNodeName) {
    if (queryNodeName.empty() || instanceNodeName.empty()) return false;
    if (instanceNodeName == queryNodeName) return true;
    const std::string matPrefix = queryNodeName + "_mat_";
    return instanceNodeName.rfind(matPrefix, 0) == 0;
}

} // namespace detail
} // namespace VulkanRT

// Unqualified spelling kept so the ~20 existing call sites need no edit.
using VulkanRT::detail::matchesNodeNameForInstance;
