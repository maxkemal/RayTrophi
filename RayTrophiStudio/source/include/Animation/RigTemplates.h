#pragma once
#include "Animation/NodeHierarchy.h"
#include "Animation/RigAnatomy.h"
namespace RigAuthoring {
struct RigTemplateInfo {
    std::string id,label,family;
    size_t joint_count=0;
    float default_height=1.8f;
    int version=1;
};
const std::vector<RigTemplateInfo>& rigTemplateCatalogue();
bool buildRigTemplate(const std::string& id,const std::string& character,float height,
                      RayTrophi::NodeHierarchy&,RigAnatomy&,std::string& error);
}
