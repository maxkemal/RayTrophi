#include "Animation/RigBinding.h"
#include "Animation/RigMeshSpace.h"
#include "Animation/RigBindingScope.h"
#include "Animation/RigBindMath.h"
#include "Animation/RigPreflight.h"
#include "Animation/RigPosePreview.h"
#include "Animation/RigWeights.h"
#include "Animation/RigSerialization.h"
#include "TriangleMesh.h"
#include "Transform.h"
#include "OzzRuntime.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_set>
namespace RigAuthoring {
namespace {
struct Token {
    uint64_t hash=1469598103934665603ull;
    void word(uint32_t value){hash^=value;hash*=1099511628211ull;}
    void size(uint64_t value){word(static_cast<uint32_t>(value));word(static_cast<uint32_t>(value>>32));}
    void number(float value){uint32_t bits;std::memcpy(&bits,&value,sizeof(bits));word(bits);}
    void name(const std::string& value){size(value.size());for(unsigned char c:value)word(c);}
    void matrix(const Matrix4x4& value){for(int r=0;r<4;++r)for(int c=0;c<4;++c)number(value.m[r][c]);}
    std::string str()const{return std::to_string(hash);}
};
bool finite(const Vec3& p){return std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.z);}
Vec3 position(const Matrix4x4& matrix){return Vec3(matrix.m[0][3],matrix.m[1][3],matrix.m[2][3]);}
bool build(const SceneData& scene,const std::string& character,const std::string& target,bool confirmed,RigBindState& output,nlohmann::json& report,std::string& error) {
    report=nullptr;error.clear();
    if(!confirmed){error="rig_bind_axes_unconfirmed";return false;}
    const SceneData::ImportedModelContext* model=nullptr;
    for(const auto& context:scene.importedModelContexts)if(context.importName==character) {
        if(model){error="ambiguous_character";return false;}model=&context;
    }
    if(!model){error="unknown_character";return false;}
    if(!model->authoringOwned){error="rig_not_owned";return false;}
    if(!model->rigBoundMeshes.empty()){error="rig_already_bound";return false;}
    if(!model->members.empty() || model->weightedBoneCount){error="rig_edit_requires_unskinned";return false;}
    if(hasFlatSkinReferences(scene,model->nodeHierarchy)){error="rig_edit_requires_unskinned";return false;}
    if(model->rigRevision==std::numeric_limits<uint64_t>::max()){error="rig_revision_overflow";return false;}
    if(model->nodeHierarchy.size()>4096){error="rig_bind_limit";return false;}
    serializeRigHierarchy(model->nodeHierarchy);
    if(!validateRigAnatomy(model->rigAnatomy,model->nodeHierarchy,error))return false;
    Matrix4x4 actor;
    if(!deserializeRigPlacement(serializeRigPlacement(model->rigSceneTransform),actor,error))return false;
    Matrix4x4 inverseActor;if(!bindAffineInverse(actor,inverseActor)){error="rig_bind_invalid_transform";return false;}
    std::vector<std::shared_ptr<TriangleMesh>> parts;
    if(!resolveFitParts(scene,target,parts,error))return false;
    if(parts.size()>256){error="rig_bind_limit";return false;}
    std::vector<PreviewJoint> joints;
    if(!sampleRigPose(model->nodeHierarchy,nullptr,0,joints,error))return false;
    std::unordered_set<std::string> nonDeforming;
    for(const auto& role:model->rigAnatomy.roles)if(role.role=="root")nonDeforming.insert(role.bone);
    std::unordered_set<unsigned int> indices;
    for(const auto& node:model->nodeHierarchy.nodes) {
        const auto found=scene.boneData.boneNameToIndex.find(node.uniqueName);
        if(found==scene.boneData.boneNameToIndex.end() || found->second>static_cast<unsigned int>(std::numeric_limits<int>::max()) ||
           !indices.insert(found->second).second){error="rig_bind_invalid_indices";return false;}
        for(const auto& entry:scene.boneData.boneNameToIndex)if(entry.second==found->second && entry.first!=node.uniqueName){error="rig_bind_invalid_indices";return false;}
    }
    std::vector<SkinSegment> segments;
    Token rigToken;rigToken.name(character);rigToken.matrix(actor);
    for(size_t i=0;i<joints.size();++i) {
        const auto& node=model->nodeHierarchy.nodes[i];rigToken.name(node.uniqueName);rigToken.matrix(node.localBind);
        rigToken.word(scene.boneData.boneNameToIndex.at(node.uniqueName));rigToken.size(static_cast<uint64_t>(node.parent+1));
        if(node.parent<0 || nonDeforming.count(joints[static_cast<size_t>(node.parent)].name))continue;
        SkinSegment segment{static_cast<int>(scene.boneData.boneNameToIndex.at(joints[static_cast<size_t>(node.parent)].name)),position(joints[static_cast<size_t>(node.parent)].world),position(joints[i].world)};
        if(segmentDistanceSquared(segment.end,{segment.bone,segment.start,segment.start})<=0){error="rig_bind_zero_length_segment";return false;}
        segments.push_back(segment);
    }
    std::vector<std::string> rootNames(nonDeforming.begin(),nonDeforming.end());
    std::sort(rootNames.begin(),rootNames.end());for(const auto& name:rootNames)rigToken.name(name);
    if(segments.empty()){error="rig_bind_requires_segments";return false;}
    Token meshToken;meshToken.name(target);meshToken.size(parts.size());
    const float inf=std::numeric_limits<float>::infinity();Vec3 lo(inf,inf,inf),hi(-inf,-inf,-inf);
    size_t vertexCount=0;std::unordered_set<const TriangleMesh*> selected;
    std::vector<Matrix4x4> conversions;std::vector<nlohmann::json> geometryReports;
    for(const auto& part:parts) {
        if(part->terrain_id>=0){error="rig_bind_requires_static_mesh";return false;}
        if(part->nodeName.empty() || part->nodeName.size()>1024){error="rig_bind_invalid_mesh_name";return false;}
        if(explicitMeshRig(scene,part->nodeName)){error="mesh_already_bound";return false;}
        if(scene.isEditorPendingDeleteObjectName(part->nodeName)){error="rig_bind_mesh_pending_delete";return false;}
        const auto modifier=scene.mesh_modifiers.find(part->nodeName);
        if((modifier!=scene.mesh_modifiers.end() && !modifier->second.modifiers.empty()) || scene.geometry_node_graphs.count(part->nodeName)){error="rig_bind_requires_static_mesh";return false;}
        // Source node animation must not keep driving a newly bound part.
        for(const auto& context:scene.importedModelContexts)if(context.importName!=character && meshBelongsToRig(scene,context.importName,*part)) {
            if(context.hasAnimation){error="rig_bind_source_has_animation";return false;}
            for(const auto& clip:scene.animationDataList)if(clip && clip->modelName==context.importName){error="rig_bind_source_has_animation";return false;}
        }
        nlohmann::json pre;if(!preflightMesh(scene,part->nodeName,pre,error))return false;
        if(pre["has_skin_weights"].get<bool>()){error="rig_bind_requires_unskinned_mesh";return false;}
        if(pre["nonfinite_vertices"].get<size_t>() || pre["invalid_triangles"].get<size_t>() || pre["trailing_indices"].get<size_t>() ||
           pre["triangle_count"].get<size_t>()==pre["degenerate_triangles"].get<size_t>()){error="rig_bind_invalid_geometry";return false;}
        const auto& g=*part->geometry;
        if(g.delta_count()){error="rig_bind_requires_static_mesh";return false;}
        const auto* normals=meshSpaceNormals(g);
        if(!normals || meshSpaceNormalCount(g)<part->num_vertices()){error="rig_bind_requires_normals";return false;}
        const auto sourceTransform=part->transform?part->transform->base:Matrix4x4::identity();
        if(part->transform) {
            for(int r=0;r<4;++r)for(int c=0;c<4;++c)
                if(std::fabs(part->transform->current.m[r][c]-(r==c?1.f:0.f))>1e-6f || !std::isfinite(part->transform->current.m[r][c])){error="rig_bind_requires_static_mesh";return false;}
        }
        Matrix4x4 sourceInverse;if(!bindAffineInverse(sourceTransform,sourceInverse)){error="rig_bind_invalid_transform";return false;}
        const auto conversion=inverseActor*sourceTransform;conversions.push_back(conversion);
        meshToken.name(part->nodeName);meshToken.size(part->num_vertices());meshToken.matrix(sourceTransform);
        const auto* positions=meshSpacePositions(g);
        for(size_t i=0;i<part->num_vertices();++i) {
            if(!finite(normals[i]) || double(normals[i].x)*normals[i].x+double(normals[i].y)*normals[i].y+double(normals[i].z)*normals[i].z<=0){error="rig_bind_invalid_normals";return false;}
            const auto p=conversion.transform_point(positions[i]);if(!finite(p)){error="rig_bind_invalid_transform";return false;}
            const auto world=sourceTransform.transform_point(positions[i]);const auto restored=actor.transform_point(p);
            if(!finite(world) || !finite(restored) ||
               std::fabs(double(restored.x)-world.x)>1e-4*std::max(1.,std::fabs(double(world.x))) ||
               std::fabs(double(restored.y)-world.y)>1e-4*std::max(1.,std::fabs(double(world.y))) ||
               std::fabs(double(restored.z)-world.z)>1e-4*std::max(1.,std::fabs(double(world.z)))){error="rig_bind_invalid_transform";return false;}
            lo.x=std::min(lo.x,p.x);lo.y=std::min(lo.y,p.y);lo.z=std::min(lo.z,p.z);hi.x=std::max(hi.x,p.x);hi.y=std::max(hi.y,p.y);hi.z=std::max(hi.z,p.z);
            for(float value:{positions[i].x,positions[i].y,positions[i].z,normals[i].x,normals[i].y,normals[i].z})meshToken.number(value);
        }
        meshToken.size(g.indices.size());for(auto index:g.indices)meshToken.word(index);
        if(part->num_vertices()>2000000-vertexCount){error="rig_bind_limit";return false;}vertexCount+=part->num_vertices();
        selected.insert(part.get());geometryReports.push_back(std::move(pre));
    }
    if(vertexCount>100000000/segments.size()){error="rig_bind_limit";return false;}
    const double extent=std::max({double(hi.x)-lo.x,double(hi.y)-lo.y,double(hi.z)-lo.z});
    if(!std::isfinite(extent) || extent<=0){error="rig_bind_invalid_geometry";return false;}
    const double floor=std::max(extent*1e-4,1e-9);
    RigBindState staged;staged.rig.bones=scene.boneData;staged.rig.model=*model;staged.view=scene.rigView;
    auto& bound = staged.rig.model;
    bound.rigBoundMeshes.clear();
    bound.members.clear();
    bound.globalInverseTransform = Matrix4x4::identity();
    bound.rigWeightAlgorithm = "nearest_segment";
    bound.rigEnvelopeProfiles.clear();
    staged.rig.bones.perModelInverses[character]=Matrix4x4::identity();
    for(size_t i=0;i<joints.size();++i) {
        Matrix4x4 inverse;if(!bindAffineInverse(joints[i].world,inverse)){error="rig_bind_invalid_rest";return false;}
        staged.rig.bones.boneOffsetMatrices[joints[i].name]=inverse;
    }
    nlohmann::json partReports=nlohmann::json::array();size_t maxInfluences=0;double minSum=1,maxSum=0,totalDistance=0,maxDistance=0;
    std::unordered_set<int> weighted;
    const auto commonTransform=std::make_shared<Transform>(actor);
    for(size_t index=0;index<parts.size();++index) {
        const auto& part=parts[index];const auto& source=*part->geometry;
        auto geometry=std::make_shared<DNA::GeometryDetail>(source);
        geometry->add_attribute<Vec3>("P_orig");geometry->add_attribute<Vec3>("N_orig");
        auto* p=geometry->get_positions_mut();auto* n=geometry->get_normals_mut();
        auto* bindP=geometry->get_attribute_data_mut<Vec3>("P_orig");auto* bindN=geometry->get_attribute_data_mut<Vec3>("N_orig");
        const auto* sourceP=meshSpacePositions(source);const auto* sourceN=meshSpaceNormals(source);
        Matrix4x4 inverse;if(!bindAffineInverse(conversions[index],inverse)){error="rig_bind_invalid_transform";return false;}
        const auto normalTransform=inverse.transpose();geometry->skin_weights.assign(part->num_vertices(),{});
        nlohmann::json samples=nlohmann::json::array();const size_t stride=std::max(size_t(1),(part->num_vertices()+15)/16);
        for(size_t vertex=0;vertex<part->num_vertices();++vertex) {
            p[vertex]=bindP[vertex]=conversions[index].transform_point(sourceP[vertex]);
            const auto normal=normalTransform.transform_vector(sourceN[vertex]);
            const double length=std::sqrt(double(normal.x)*normal.x+double(normal.y)*normal.y+double(normal.z)*normal.z);
            if(!finite(normal) || !std::isfinite(length) || length<=0){error="rig_bind_invalid_normals";return false;}
            n[vertex]=bindN[vertex]=Vec3(static_cast<float>(double(normal.x)/length),static_cast<float>(double(normal.y)/length),static_cast<float>(double(normal.z)/length));
            double distance;auto weights=distanceSkinWeights(p[vertex],segments,floor,distance);
            if(weights.empty()){error="rig_bind_empty_weights";return false;}
            double sum=0;for(const auto& weight:weights){sum+=weight.second;weighted.insert(weight.first);}
            minSum=std::min(minSum,sum);maxSum=std::max(maxSum,sum);maxInfluences=std::max(maxInfluences,weights.size());totalDistance+=distance;maxDistance=std::max(maxDistance,distance);
            if(vertex%stride==0)samples.push_back({{"vertex",vertex},{"weights",weights}});
            geometry->skin_weights[vertex]=std::move(weights);
        }
        geometry->last_skinned_pose_hash=0;staged.parts.push_back({part,std::move(geometry),commonTransform});
        bound.rigBoundMeshes.push_back(part->nodeName);bound.members.push_back(part);
        partReports.push_back({{"mesh",part->nodeName},{"vertex_count",part->num_vertices()},{"degenerate_triangles",geometryReports[index]["degenerate_triangles"]},{"samples",samples}});
    }
    for(const auto& node:bound.nodeHierarchy.nodes) {
        const int index=static_cast<int>(staged.rig.bones.boneNameToIndex.at(node.uniqueName));
        if(weighted.count(index))staged.rig.bones.weightedBoneNames.insert(node.uniqueName);
    }
    staged.rig.bones.rebuildReverseLookup();bound.rebuildSkeletonRepresentation(staged.rig.bones);
    std::vector<std::shared_ptr<AnimationData>> characterClips;
    for(const auto& clip:scene.animationDataList)
        if(clip && clip->modelName==character)characterClips.push_back(clip);
    bound.ozzAnimationSet=OzzRuntime::buildStubAnimationSet(character,staged.rig.bones,characterClips);
    bound.animator=std::make_shared<AnimationController>();
    bound.animator->registerClips(characterClips);
    bound.hasAnimation=!characterClips.empty();
    bound.rigJointGlobals.clear();bound.rigPoseSource="bind";bound.restPoseApplied=false;bound.rigPoseViewCpuApplied=false;bound.rigPoseViewCpuRestorePending=false;++bound.rigRevision;
    for(const auto& context:scene.importedModelContexts)if(context.importName!=character) {
        auto members=context.members;members.erase(std::remove_if(members.begin(),members.end(),[&](const auto& member){return selected.count(dynamic_cast<const TriangleMesh*>(member.get()))>0;}),members.end());
        if(members.size()!=context.members.size())staged.memberships.push_back({context.importName,std::move(members)});
    }
    staged.view.character=character;staged.view.bone=bound.nodeHierarchy.nodes.front().uniqueName;
    if(staged.view.edit_character==character){staged.view.edit_mode=false;staged.view.edit_character.clear();}
    staged.view.pose_views[character]="rest";staged.view.pose_view_dirty.insert(character);
    size_t outside=0;const double tolerance=extent*1e-5;
    for(const auto& joint:joints){const auto p=position(joint.world);if(p.x<lo.x-tolerance||p.x>hi.x+tolerance||p.y<lo.y-tolerance||p.y>hi.y+tolerance||p.z<lo.z-tolerance||p.z>hi.z+tolerance)++outside;}
    report={{"character",character},{"mesh",target},{"axes_confirmed",confirmed},{"rig_revision",model->rigRevision},
        {"rig_token",rigToken.str()},{"mesh_token",meshToken.str()},{"algorithm","nearest_segment"},{"part_count",parts.size()},
        {"vertex_count",vertexCount},{"parts",partReports},{"segment_count",segments.size()},{"weighted_bone_count",weighted.size()},
        {"max_influences",maxInfluences},{"min_weight_sum",minSum},{"max_weight_sum",maxSum},{"unweighted_vertices",0},
        {"outside_bounds_joints",outside},{"mean_nearest_distance_rig",totalDistance/vertexCount},{"max_nearest_distance_rig",maxDistance},
        {"surface_interior_verified",false},{"visibility_verified",false},{"alignment_verified",false},{"can_bind",true}};
    output=std::move(staged);return true;
}
}
bool previewMeshBinding(const SceneData& scene,const std::string& character,const std::string& mesh,bool confirmed,nlohmann::json& output,std::string& error) {
    RigBindState state;return build(scene,character,mesh,confirmed,state,output,error);
}
bool stageMeshBinding(const SceneData& scene,const std::string& character,const std::string& mesh,const nlohmann::json& preview,RigBindState& output,std::string& error) {
    if(!preview.is_object() || !preview.contains("rig_revision") || !preview["rig_revision"].is_number_unsigned() ||
       !preview.contains("rig_token") || !preview["rig_token"].is_string() || !preview.contains("mesh_token") || !preview["mesh_token"].is_string() ||
       !preview.contains("axes_confirmed") || !preview["axes_confirmed"].is_boolean()){error="rig_bind_invalid_preview";return false;}
    nlohmann::json verified;RigBindState staged;
    if(!build(scene,character,mesh,preview["axes_confirmed"].get<bool>(),staged,verified,error))return false;
    if(preview["rig_revision"]!=verified["rig_revision"] || preview["rig_token"]!=verified["rig_token"] || preview["mesh_token"]!=verified["mesh_token"]){error="rig_bind_stale_preview";return false;}
    output=std::move(staged);return true; // Ignore caller-supplied weights and can_bind.
}
bool stageMeshUnbinding(const SceneData& scene,const std::string& character,
                        RigBindState& output,std::string& error) {
    error.clear();
    const SceneData::ImportedModelContext* model = nullptr;
    for(const auto& candidate : scene.importedModelContexts) {
        if(candidate.importName != character) {
            continue;
        }
        if(model) {
            error = "ambiguous_character";
            return false;
        }
        model = &candidate;
    }
    if(!model) {
        error = "unknown_character";
        return false;
    }
    if(!model->authoringOwned) {
        error = "rig_not_owned";
        return false;
    }
    if(model->rigBoundMeshes.empty()) {
        error = "rig_not_bound";
        return false;
    }
    if(model->rigRevision == std::numeric_limits<uint64_t>::max()) {
        error = "rig_revision_overflow";
        return false;
    }

    RigBindState staged;
    staged.rig.bones=scene.boneData;
    staged.rig.model=*model;
    staged.view=scene.rigView;
    auto& unbound=staged.rig.model;

    for(const auto& meshName : model->rigBoundMeshes) {
        std::shared_ptr<TriangleMesh> mesh;
        for(const auto& object : scene.world.objects) {
            auto candidate = std::dynamic_pointer_cast<TriangleMesh>(object);
            if(candidate && candidate->nodeName == meshName) {
                if(mesh && mesh != candidate) {
                    error = "ambiguous_mesh_name";
                    return false;
                }
                mesh = std::move(candidate);
            }
        }
        if(!mesh || !mesh->geometry) {
            continue;
        }
        auto geometry = std::make_shared<DNA::GeometryDetail>(*mesh->geometry);
        const auto* bindP = geometry->get_positions_orig();
        const auto* bindN = geometry->get_normals_orig();
        auto* positions = geometry->get_positions_mut();
        auto* normals = geometry->get_normals_mut();
        const size_t vertexCount = geometry->get_vertex_count();
        if(bindP && positions) {
            for(size_t i = 0; i < vertexCount; ++i) {
                positions[i] = bindP[i];
            }
        }
        if(bindN && normals) {
            for(size_t i = 0; i < vertexCount; ++i) {
                normals[i] = bindN[i];
            }
        }
        geometry->skin_weights.clear();
        geometry->last_skinned_pose_hash=0;
        staged.parts.push_back({mesh,std::move(geometry),mesh->transform});
    }

    for(const auto& node : unbound.nodeHierarchy.nodes) {
        staged.rig.bones.weightedBoneNames.erase(node.uniqueName);
    }
    staged.rig.bones.rebuildReverseLookup();
    unbound.rigBoundMeshes.clear();
    unbound.members.clear();
    unbound.rigWeightAlgorithm.clear();
    unbound.rigEnvelopeProfiles.clear();
    unbound.weightedBoneCount=0;
    unbound.rigJointGlobals.clear();
    unbound.rigPoseSource = "bind";
    unbound.restPoseApplied = false;
    unbound.rigPoseViewCpuApplied = false;
    unbound.rigPoseViewCpuRestorePending = false;
    ++unbound.rigRevision;
    unbound.rebuildSkeletonRepresentation(staged.rig.bones);
    staged.view.pose_views[character] = "rest";
    staged.view.pose_view_dirty.insert(character);
    output = std::move(staged);
    return true;
}
bool meshBindingInfo(const SceneData& scene, const std::string& character,
                     nlohmann::json& output, std::string& error) {
    output = nullptr;
    error.clear();
    for (const auto& model : scene.importedModelContexts) {
        if (model.importName != character)
            continue;
        nlohmann::json parts = nlohmann::json::array();
        for (const auto& name : model.rigBoundMeshes) {
            bool present = false;
            for (const auto& object : scene.world.objects) {
                auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object);
                if (mesh && mesh->nodeName == name) {
                    present = true;
                    break;
                }
            }
            parts.push_back({{"mesh", name}, {"present", present}});
        }
        const auto algorithm = model.rigWeightAlgorithm.empty()
                                   ? "nearest_segment"
                                   : model.rigWeightAlgorithm;
        output = {{"character", character},
                  {"bound", !model.rigBoundMeshes.empty()},
                  {"algorithm", model.rigBoundMeshes.empty()
                                    ? nlohmann::json(nullptr)
                                    : nlohmann::json(algorithm)},
                  {"parts", parts},
                  {"rig_revision", model.rigRevision}};
        if (model.rigWeightAlgorithm == "anatomical_capsule_v1") {
            output["envelope_settings"] = {
                {"torso_radius", model.rigEnvelopeTorsoRadius},
                {"limb_radius", model.rigEnvelopeLimbRadius},
                {"extremity_radius", model.rigEnvelopeExtremityRadius},
                {"falloff", model.rigEnvelopeFalloff}};
        }
        return true;
    }
    error = "unknown_character";
    return false;
}
}
