#pragma once
#include <string>
namespace rtimport {
// Session-wide choice, sampled once at dispatch. A failed setting change leaves
// the current value intact. No fallback between readers on an import failure.
std::string getFbxReader();
bool setFbxReader(const std::string& reader, std::string& error);
}
