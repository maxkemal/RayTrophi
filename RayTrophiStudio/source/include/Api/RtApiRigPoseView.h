#pragma once
#include <string>
namespace rtapi {
struct Result;
Result setRigPoseView(const std::string& character,const std::string& mode);
Result getRigPoseView(const std::string& character,std::string& mode,std::string& effectiveMode);
}
