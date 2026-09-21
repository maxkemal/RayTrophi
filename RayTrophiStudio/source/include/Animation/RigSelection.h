#pragma once
#include "Animation/RigView.h"
namespace RigAuthoring {
std::vector<std::string> selectedBones(const SceneData&);
bool isBoneSelected(const ViewState&, const std::string& character, const std::string& bone);
bool selectBones(SceneData&, const std::string& character, const std::vector<std::string>& bones,
                 const std::string& active, const std::string& mode, std::string& error,
                 const std::string& anchor = "");
bool setSelectionPivot(SceneData&, const std::string& mode, std::string& error);
void exchangeSelection(ViewState&, ViewState&);
void releaseIKHandleSelection(ViewState&);
}
