#pragma once
#include "Animation/RigEditing.h"
#include <memory>
class TriangleMesh;
class Transform;
namespace DNA { class GeometryDetail; }
namespace RigAuthoring {
struct BindPartState {
    std::shared_ptr<TriangleMesh> mesh;
    std::shared_ptr<DNA::GeometryDetail> geometry;
    std::shared_ptr<Transform> transform;
};
struct BindMembershipState {std::string character;std::vector<std::shared_ptr<Hittable>> members;};
struct RigBindState {
    RigEditState rig;
    ViewState view;
    std::vector<BindPartState> parts;
    std::vector<BindMembershipState> memberships;
};
bool previewMeshBinding(const SceneData&,const std::string& character,const std::string& mesh,bool axesConfirmed,nlohmann::json&,std::string& error);
bool stageMeshBinding(const SceneData&,const std::string& character,const std::string& mesh,const nlohmann::json& preview,RigBindState&,std::string& error);
bool stageMeshUnbinding(const SceneData&,const std::string& character,RigBindState&,std::string& error);
bool meshBindingInfo(const SceneData&,const std::string& character,nlohmann::json&,std::string& error);
}
