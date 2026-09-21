#include "Api/RtApi.h"
#include "Import/ImportSettings.h"
namespace rtapi {
std::string getFbxReader() { return rtimport::getFbxReader(); }
Result setFbxReader(const std::string& reader) {
    std::string error;
    return rtimport::setFbxReader(reader, error) ? Result::success() : Result::fail(error);
}
}
