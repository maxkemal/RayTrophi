#include "Import/ImportSettings.h"
#include <atomic>
namespace rtimport {
namespace { std::atomic<bool> useUfbx{false}; }
std::string getFbxReader() { return useUfbx.load() ? "ufbx" : "assimp"; }
bool setFbxReader(const std::string& reader, std::string& error) {
    error.clear();
    if (reader != "assimp" && reader != "ufbx") {
        error = "reader must be 'assimp' or 'ufbx' (static FBX increment)";
        return false;
    }
    useUfbx.store(reader == "ufbx");
    return true;
}
}
