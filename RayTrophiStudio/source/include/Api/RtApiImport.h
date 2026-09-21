#pragma once
#include <string>
namespace rtapi {
struct Result;
std::string getFbxReader();
Result setFbxReader(const std::string& reader);
}
