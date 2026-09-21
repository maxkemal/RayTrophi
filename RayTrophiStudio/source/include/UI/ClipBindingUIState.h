#pragma once
#include "Animation/ClipBinding.h"
struct UIContext;
namespace RigUI {
struct ClipBindingUIState {
    std::string source, clip, message;
    RigAuthoring::ClipBindingReport report;
    std::map<std::string, std::string> nodeMap;
    bool restBasis=false;
    float translationScale=1.f;
    bool previewed=false;
};
ClipBindingUIState& clipBindingState(const std::string& target);
void drawClipBindingContents(UIContext&, const std::string& target, bool manual);
}
