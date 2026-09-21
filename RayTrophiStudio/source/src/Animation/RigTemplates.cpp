#include "Animation/RigTemplates.h"
#include <cmath>
#include <utility>

namespace RigAuthoring {
namespace {
struct Joint {std::string name;int parent;Vec3 position;std::string role;};
struct Definition {
    std::string id,label,family;
    float defaultHeight=1.8f;
    int version=1;
    std::vector<Joint> joints;
    RigAnatomy anatomy;
    int index(const std::string& name) const {
        for(size_t i=0;i<joints.size();++i)if(joints[i].name==name)return static_cast<int>(i);
        return -1;
    }
    void add(const std::string& name,const std::string& parent,Vec3 p,const std::string& role="") {
        joints.push_back({name,parent.empty()?-1:index(parent),p,role});
    }
    void pair(const std::string& left,const std::string& right){anatomy.symmetry.push_back({left,right});}
    void chain(const std::string& name,std::vector<std::string> bones){anatomy.chains.push_back({name,std::move(bones)});}
    void fit(const std::string& bone,const std::string& start,const std::string& end,float position) {
        anatomy.fitRules.push_back({bone,start,end,position});
    }
};
Definition basic(bool chain) {
    Definition d;d.id=chain?"chain3":"root";d.label=chain?"Three-joint chain":"Root";d.family="custom";
    d.add("Root","",Vec3(0,0,0));
    if(chain){d.add("Joint1","Root",Vec3(0,.5f,0));d.add("Joint2","Joint1",Vec3(0,1,0));}
    return d;
}
Definition humanoid() {
    Definition d;d.id="humanoid";d.label="Humanoid (T rest)";d.family="humanoid";d.version=3;
    // Height-normalized fitting seed with a mild sagittal spine profile,
    // forward knee/elbow hints and symmetric level arms. This is not a fitted bind pose.
    // Joint axes and IK bend preferences belong to the later control layer.
    d.add("Root","",Vec3(0,0,0),"root");d.add("Pelvis","Root",Vec3(0,.53f,0),"pelvis");
    d.add("Spine","Pelvis",Vec3(0,.62f,.008f),"spine.lower");d.add("Chest","Spine",Vec3(0,.735f,-.012f),"spine.upper");
    d.add("Neck","Chest",Vec3(0,.835f,0),"neck");d.add("Head","Neck",Vec3(0,.875f,.012f),"head");
    d.add("HeadEnd","Head",Vec3(0,1,.012f),"head.tip");
    d.chain("spine",{"Pelvis","Spine","Chest","Neck","Head","HeadEnd"});
    for(int side=0;side<2;++side) {
        const std::string s=side==0?"Left":"Right",role=side==0?"left":"right";
        const float x=side==0?1.f:-1.f;
        d.add(s+"Clavicle","Chest",Vec3(x*.035f,.80f,-.006f),role+"_arm.clavicle");
        d.add(s+"UpperArm",s+"Clavicle",Vec3(x*.115f,.80f,-.012f),role+"_arm.upper");
        d.add(s+"Forearm",s+"UpperArm",Vec3(x*.285f,.80f,-.004f),role+"_arm.lower");
        d.add(s+"Hand",s+"Forearm",Vec3(x*.425f,.80f,-.012f),role+"_arm.hand");
        d.add(s+"HandEnd",s+"Hand",Vec3(x*.50f,.80f,-.012f),role+"_arm.hand_tip");
        d.chain(role+"_arm",{s+"Clavicle",s+"UpperArm",s+"Forearm",s+"Hand",s+"HandEnd"});
        d.add(s+"Thigh","Pelvis",Vec3(x*.055f,.515f,-.004f),role+"_leg.upper");
        d.add(s+"Shin",s+"Thigh",Vec3(x*.055f,.285f,.014f),role+"_leg.lower");
        d.add(s+"Foot",s+"Shin",Vec3(x*.055f,.045f,0),role+"_leg.ankle");
        d.add(s+"Toe",s+"Foot",Vec3(x*.055f,.02f,.095f),role+"_leg.toe");
        d.add(s+"ToeEnd",s+"Toe",Vec3(x*.055f,.02f,.14f),role+"_leg.toe_tip");
        d.chain(role+"_leg",{s+"Thigh",s+"Shin",s+"Foot",s+"Toe",s+"ToeEnd"});
    }
    for(const char* name:{"Clavicle","UpperArm","Forearm","Hand","HandEnd","Thigh","Shin","Foot","Toe","ToeEnd"})
        d.pair(std::string("Left")+name,std::string("Right")+name);
    return d;
}

void addHumanoidFinger(Definition& definition, const std::string& side,
                       const std::string& roleSide, float direction,
                       const std::string& fingerName, const std::string& fingerRole,
                       const Vec3 positions[4]) {
    const std::string prefix = side + fingerName;
    const std::string rolePrefix = roleSide + "_hand." + fingerRole;
    const char* suffixes[] = {"1", "2", "3", "End"};
    const char* roleSuffixes[] = {"01", "02", "03", "tip"};
    std::vector<std::string> chain = {side + "Hand"};

    std::string parent = side + "Hand";
    for (int index = 0; index < 4; ++index) {
        const std::string name = prefix + suffixes[index];
        Vec3 position = positions[index];
        position.x *= direction;
        definition.add(name, parent, position, rolePrefix + "." + roleSuffixes[index]);
        chain.push_back(name);
        parent = name;
    }
    definition.chain(roleSide + "_" + fingerRole, std::move(chain));
}

void addHumanoidHandControls(Definition& definition, const std::string& side,
                             const std::string& roleSide, float direction) {
    RigDrivenControl curl;
    curl.id = roleSide + "_hand.curl";
    curl.label = "Curl";
    curl.group = roleSide + "_hand";
    curl.anchor = side + "Hand";
    curl.side = roleSide;
    curl.shape = "hand";
    const float curlWeights[] = {.72f, .88f, 1.f};
    for (const char* finger : {"Thumb", "Index", "Middle", "Ring", "Pinky"}) {
        const float degrees = std::string(finger) == "Thumb" ? 58.f : 82.f;
        for (int joint = 1; joint <= 3; ++joint) {
            curl.drivers.push_back({side + finger + std::to_string(joint), Vec3(0, 0, 1),
                                    -direction * degrees * curlWeights[joint - 1]});
        }
    }
    definition.anatomy.drivenControls.push_back(std::move(curl));

    RigDrivenControl spread;
    spread.id = roleSide + "_hand.spread";
    spread.label = "Spread";
    spread.group = roleSide + "_hand";
    spread.anchor = side + "Hand";
    spread.side = roleSide;
    spread.shape = "hand";
    const char* spreadFingers[] = {"Index", "Middle", "Ring", "Pinky"};
    const float spreadWeights[] = {1.f, .25f, -.35f, -1.f};
    for (int finger = 0; finger < 4; ++finger) {
        spread.drivers.push_back({side + spreadFingers[finger] + "1", Vec3(0, 1, 0),
                                  -direction * 18.f * spreadWeights[finger]});
    }
    definition.anatomy.drivenControls.push_back(std::move(spread));

    RigDrivenControl thumb;
    thumb.id = roleSide + "_hand.thumb";
    thumb.label = "Thumb opposition";
    thumb.group = roleSide + "_hand";
    thumb.anchor = side + "Hand";
    thumb.side = roleSide;
    thumb.shape = "hand";
    for (int joint = 1; joint <= 3; ++joint) {
        thumb.drivers.push_back({side + "Thumb" + std::to_string(joint), Vec3(0, 1, 0),
                                 -direction * (joint == 1 ? 30.f : 8.f)});
        thumb.drivers.push_back({side + "Thumb" + std::to_string(joint), Vec3(0, 0, 1),
                                 -direction * 52.f * curlWeights[joint - 1]});
    }
    definition.anatomy.drivenControls.push_back(std::move(thumb));
}

Definition detailedHumanoid() {
    Definition definition;
    definition.id = "humanoid_detailed";
    definition.label = "Humanoid Detailed";
    definition.family = "humanoid";
    definition.version = 2;

    definition.add("Root", "", Vec3(0, 0, 0), "root");
    definition.add("Pelvis", "Root", Vec3(0, .53f, 0), "pelvis");
    definition.add("Spine01", "Pelvis", Vec3(0, .585f, .006f), "spine.lower");
    definition.add("Spine02", "Spine01", Vec3(0, .64f, .004f), "spine.middle");
    definition.add("Spine03", "Spine02", Vec3(0, .695f, -.004f), "spine.chest");
    definition.add("Chest", "Spine03", Vec3(0, .735f, -.012f), "spine.upper");
    definition.add("Neck01", "Chest", Vec3(0, .79f, -.004f), "neck");
    definition.add("Neck02", "Neck01", Vec3(0, .835f, .004f), "neck.upper");
    definition.add("Head", "Neck02", Vec3(0, .875f, .012f), "head");
    definition.add("HeadEnd", "Head", Vec3(0, 1, .012f), "head.tip");
    definition.chain("spine", {"Pelvis", "Spine01", "Spine02", "Spine03", "Chest",
                                   "Neck01", "Neck02", "Head", "HeadEnd"});
    definition.fit("Spine01", "Pelvis", "Chest", .27f);
    definition.fit("Spine02", "Pelvis", "Chest", .54f);
    definition.fit("Spine03", "Pelvis", "Chest", .81f);
    definition.fit("Neck01", "Chest", "Head", .40f);
    definition.fit("Neck02", "Chest", "Head", .72f);

    for (int sideIndex = 0; sideIndex < 2; ++sideIndex) {
        const std::string side = sideIndex == 0 ? "Left" : "Right";
        const std::string roleSide = sideIndex == 0 ? "left" : "right";
        const float direction = sideIndex == 0 ? 1.f : -1.f;
        definition.add(side + "Clavicle", "Chest", Vec3(direction * .035f, .80f, -.006f),
                       roleSide + "_arm.clavicle");
        definition.add(side + "UpperArm", side + "Clavicle",
                       Vec3(direction * .115f, .80f, -.012f), roleSide + "_arm.upper");
        definition.add(side + "Forearm", side + "UpperArm",
                       Vec3(direction * .285f, .80f, -.004f), roleSide + "_arm.lower");
        definition.add(side + "Hand", side + "Forearm",
                       Vec3(direction * .425f, .80f, -.012f), roleSide + "_arm.hand");
        definition.add(side + "HandEnd", side + "Hand",
                       Vec3(direction * .50f, .80f, -.012f), roleSide + "_arm.hand_tip");
        definition.chain(roleSide + "_arm", {side + "Clavicle", side + "UpperArm",
                                               side + "Forearm", side + "Hand",
                                               side + "HandEnd"});
        definition.add(side + "Thigh", "Pelvis", Vec3(direction * .055f, .515f, -.004f),
                       roleSide + "_leg.upper");
        definition.add(side + "Shin", side + "Thigh",
                       Vec3(direction * .055f, .285f, .014f), roleSide + "_leg.lower");
        definition.add(side + "Foot", side + "Shin", Vec3(direction * .055f, .045f, 0),
                       roleSide + "_leg.ankle");
        definition.add(side + "Toe", side + "Foot", Vec3(direction * .055f, .02f, .095f),
                       roleSide + "_leg.toe");
        definition.add(side + "ToeEnd", side + "Toe",
                       Vec3(direction * .055f, .02f, .14f), roleSide + "_leg.toe_tip");
        definition.chain(roleSide + "_leg", {side + "Thigh", side + "Shin", side + "Foot",
                                               side + "Toe", side + "ToeEnd"});
    }
    for (const char* name : {"Clavicle", "UpperArm", "Forearm", "Hand", "HandEnd",
                             "Thigh", "Shin", "Foot", "Toe", "ToeEnd"}) {
        definition.pair(std::string("Left") + name, std::string("Right") + name);
    }

    const Vec3 thumb[4] = {
        Vec3(.438f, .785f, .010f),
        Vec3(.455f, .770f, .027f),
        Vec3(.476f, .758f, .042f),
        Vec3(.497f, .750f, .052f),
    };
    const Vec3 index[4] = {
        Vec3(.445f, .806f, .018f),
        Vec3(.475f, .806f, .018f),
        Vec3(.505f, .806f, .018f),
        Vec3(.535f, .806f, .018f),
    };
    const Vec3 middle[4] = {
        Vec3(.447f, .802f, .002f),
        Vec3(.480f, .802f, .002f),
        Vec3(.514f, .802f, .002f),
        Vec3(.550f, .802f, .002f),
    };
    const Vec3 ring[4] = {
        Vec3(.444f, .798f, -.015f),
        Vec3(.475f, .798f, -.015f),
        Vec3(.505f, .798f, -.015f),
        Vec3(.535f, .798f, -.015f),
    };
    const Vec3 pinky[4] = {
        Vec3(.438f, .793f, -.030f),
        Vec3(.465f, .793f, -.030f),
        Vec3(.490f, .793f, -.030f),
        Vec3(.512f, .793f, -.030f),
    };

    for (int sideIndex = 0; sideIndex < 2; ++sideIndex) {
        const std::string side = sideIndex == 0 ? "Left" : "Right";
        const std::string roleSide = sideIndex == 0 ? "left" : "right";
        const float direction = sideIndex == 0 ? 1.f : -1.f;
        addHumanoidFinger(definition, side, roleSide, direction, "Thumb", "thumb", thumb);
        addHumanoidFinger(definition, side, roleSide, direction, "Index", "index", index);
        addHumanoidFinger(definition, side, roleSide, direction, "Middle", "middle", middle);
        addHumanoidFinger(definition, side, roleSide, direction, "Ring", "ring", ring);
        addHumanoidFinger(definition, side, roleSide, direction, "Pinky", "pinky", pinky);
        for (const char* finger : {"Thumb", "Index", "Middle", "Ring", "Pinky"}) {
            const std::string start = side + "Hand";
            const std::string end = side + finger + "End";
            definition.fit(side + finger + "1", start, end, .25f);
            definition.fit(side + finger + "2", start, end, .50f);
            definition.fit(side + finger + "3", start, end, .75f);
        }
        addHumanoidHandControls(definition, side, roleSide, direction);
    }

    for (const char* finger : {"Thumb", "Index", "Middle", "Ring", "Pinky"}) {
        for (const char* suffix : {"1", "2", "3", "End"}) {
            definition.pair(std::string("Left") + finger + suffix,
                            std::string("Right") + finger + suffix);
        }
    }
    return definition;
}

Definition quadruped() {
    Definition d;d.id="quadruped";d.label="Quadruped (standing rest)";d.family="quadruped";d.defaultHeight=1.f;d.version=2;
    d.add("Root","",Vec3(0,0,0),"root");d.add("Pelvis","Root",Vec3(0,.67f,-.32f),"pelvis");
    d.add("Spine","Pelvis",Vec3(0,.69f,-.12f),"spine.lower");d.add("Chest","Spine",Vec3(0,.70f,.25f),"spine.upper");
    d.add("Neck","Chest",Vec3(0,.82f,.43f),"neck");d.add("Head","Neck",Vec3(0,.91f,.59f),"head");
    d.add("HeadEnd","Head",Vec3(0,1,.70f),"head.tip");
    d.chain("spine",{"Pelvis","Spine","Chest","Neck","Head","HeadEnd"});
    d.add("TailBase","Pelvis",Vec3(0,.67f,-.47f),"tail.base");
    d.add("TailMid","TailBase",Vec3(0,.55f,-.70f),"tail.middle");d.add("TailEnd","TailMid",Vec3(0,.50f,-.88f),"tail.tip");
    d.chain("tail",{"TailBase","TailMid","TailEnd"});
    for(int side=0;side<2;++side)for(int limb=0;limb<2;++limb) {
        const std::string s=side==0?"Left":"Right",part=limb==0?"Front":"Hind";
        const auto prefix=s+part;const float x=side==0?.12f:-.12f;
        const std::string role=(side==0?"left_":"right_")+std::string(limb==0?"front_leg":"hind_leg");
        d.add(prefix+"Upper",limb==0?"Chest":"Pelvis",Vec3(x,limb==0?.65f:.62f,limb==0?.30f:-.32f),role+".upper");
        d.add(prefix+"Lower",prefix+"Upper",Vec3(x,limb==0?.36f:.34f,limb==0?.32f:-.44f),role+".lower");
        d.add(prefix+"Paw",prefix+"Lower",Vec3(x,0,limb==0?.30f:-.31f),role+".paw");
        d.add(prefix+"PawEnd",prefix+"Paw",Vec3(x,0,limb==0?.44f:-.17f),role+".paw_tip");
        d.chain(role,{prefix+"Upper",prefix+"Lower",prefix+"Paw",prefix+"PawEnd"});
    }
    for(const char* part:{"Front","Hind"})for(const char* joint:{"Upper","Lower","Paw","PawEnd"})
        d.pair(std::string("Left")+part+joint,std::string("Right")+part+joint);
    return d;
}
Definition insect() {
    Definition d;d.id="insect6";d.label="Insect (six legs)";d.family="insect";d.defaultHeight=.3f;d.version=2;
    d.add("Root","",Vec3(0,0,0),"root");d.add("Abdomen","Root",Vec3(0,.20f,-.30f),"abdomen");
    d.add("Thorax","Abdomen",Vec3(0,.24f,0),"thorax");d.add("Head","Thorax",Vec3(0,.27f,.22f),"head");
    d.add("HeadEnd","Head",Vec3(0,.27f,.36f),"head.tip");d.add("AbdomenEnd","Abdomen",Vec3(0,.19f,-.53f),"abdomen.tip");
    d.chain("body",{"Abdomen","Thorax","Head","HeadEnd"});
    for(int side=0;side<2;++side) {
        const std::string s=side==0?"Left":"Right",r=side==0?"left":"right";const float x=side==0?1.f:-1.f;
        d.add(s+"AntennaBase","Head",Vec3(x*.06f,.29f,.29f),r+"_antenna.base");
        d.add(s+"AntennaMid",s+"AntennaBase",Vec3(x*.12f,.38f,.39f),r+"_antenna.middle");
        d.add(s+"AntennaEnd",s+"AntennaMid",Vec3(x*.16f,.46f,.51f),r+"_antenna.tip");
        d.chain(r+"_antenna",{s+"AntennaBase",s+"AntennaMid",s+"AntennaEnd"});
        for(int limb=0;limb<3;++limb) {
            const char* part=limb==0?"Front":limb==1?"Middle":"Hind";
            const std::string prefix=s+part,role=r+"_leg"+std::to_string(limb+1);
            const float z=limb==0?.17f:limb==1?0.f:-.19f;
            const float kneeZ=limb==0?.22f:limb==1?-.03f:-.35f;
            const float tipZ=limb==0?.34f:limb==1?-.06f:-.50f;
            d.add(prefix+"Coxa",limb==2?"Abdomen":"Thorax",Vec3(x*.13f,.22f,z),role+".coxa");
            d.add(prefix+"Femur",prefix+"Coxa",Vec3(x*.30f,.26f,kneeZ),role+".femur");
            d.add(prefix+"Tibia",prefix+"Femur",Vec3(x*.45f,.12f,tipZ),role+".tibia");
            d.add(prefix+"Tarsus",prefix+"Tibia",Vec3(x*.50f,0,tipZ),role+".tarsus");
            d.chain(role,{prefix+"Coxa",prefix+"Femur",prefix+"Tibia",prefix+"Tarsus"});
        }
    }
    for(const char* joint:{"AntennaBase","AntennaMid","AntennaEnd"})d.pair(std::string("Left")+joint,std::string("Right")+joint);
    for(const char* part:{"Front","Middle","Hind"})for(const char* joint:{"Coxa","Femur","Tibia","Tarsus"})
        d.pair(std::string("Left")+part+joint,std::string("Right")+part+joint);
    // Overall scale is ground-to-highest endpoint height, including antennae.
    for(auto& joint:d.joints)joint.position=joint.position/.46f;
    return d;
}
Definition avian() {
    Definition d;d.id="avian";d.label="Avian (spread wings)";d.family="avian";d.defaultHeight=.6f;
    // Generic bird layout: paired articulated wings are independent of the leg chains.
    // This is a fitting seed, not a universal bird/bat/insect anatomy or flight solver.
    d.add("Root","",Vec3(0,0,0),"root");d.add("Pelvis","Root",Vec3(0,.43f,-.08f),"pelvis");
    d.add("Spine","Pelvis",Vec3(0,.53f,0),"spine.lower");d.add("Chest","Spine",Vec3(0,.64f,.08f),"spine.upper");
    d.add("Neck","Chest",Vec3(0,.80f,.12f),"neck");d.add("Head","Neck",Vec3(0,.91f,.19f),"head");
    d.add("HeadEnd","Head",Vec3(0,1,.19f),"head.tip");
    d.add("Beak","Head",Vec3(0,.89f,.31f),"beak.base");d.add("BeakEnd","Beak",Vec3(0,.87f,.46f),"beak.tip");
    d.chain("spine",{"Pelvis","Spine","Chest","Neck","Head","HeadEnd"});
    d.chain("beak",{"Head","Beak","BeakEnd"});
    d.add("TailBase","Pelvis",Vec3(0,.43f,-.22f),"tail.base");
    d.add("TailMid","TailBase",Vec3(0,.40f,-.40f),"tail.middle");d.add("TailEnd","TailMid",Vec3(0,.37f,-.62f),"tail.tip");
    d.chain("tail",{"TailBase","TailMid","TailEnd"});
    for(int side=0;side<2;++side) {
        const std::string s=side==0?"Left":"Right",role=side==0?"left":"right";const float x=side==0?1.f:-1.f;
        d.add(s+"WingShoulder","Chest",Vec3(x*.08f,.68f,.05f),role+"_wing.shoulder");
        d.add(s+"WingElbow",s+"WingShoulder",Vec3(x*.32f,.68f,-.04f),role+"_wing.elbow");
        d.add(s+"WingWrist",s+"WingElbow",Vec3(x*.55f,.68f,-.10f),role+"_wing.wrist");
        d.add(s+"WingTip",s+"WingWrist",Vec3(x*.78f,.68f,-.20f),role+"_wing.tip");
        d.add(s+"WingEnd",s+"WingTip",Vec3(x*.95f,.68f,-.24f),role+"_wing.end");
        d.chain(role+"_wing",{s+"WingShoulder",s+"WingElbow",s+"WingWrist",s+"WingTip",s+"WingEnd"});
        d.add(s+"Thigh","Pelvis",Vec3(x*.075f,.405f,-.035f),role+"_leg.upper");
        d.add(s+"Knee",s+"Thigh",Vec3(x*.075f,.30f,.035f),role+"_leg.knee");
        d.add(s+"Ankle",s+"Knee",Vec3(x*.075f,.115f,-.045f),role+"_leg.ankle");
        d.add(s+"Toe",s+"Ankle",Vec3(x*.075f,0,.08f),role+"_leg.toe");
        d.add(s+"ToeEnd",s+"Toe",Vec3(x*.075f,0,.20f),role+"_leg.toe_tip");
        d.chain(role+"_leg",{s+"Thigh",s+"Knee",s+"Ankle",s+"Toe",s+"ToeEnd"});
    }
    for(const char* joint:{"WingShoulder","WingElbow","WingWrist","WingTip","WingEnd","Thigh","Knee","Ankle","Toe","ToeEnd"})
        d.pair(std::string("Left")+joint,std::string("Right")+joint);
    return d;
}
const std::vector<Definition>& definitions() {
    static const std::vector<Definition> values = {
        basic(false), basic(true), humanoid(), detailedHumanoid(), quadruped(), insect(), avian()};
    return values;
}
}
const std::vector<RigTemplateInfo>& rigTemplateCatalogue() {
    static const auto catalogue=[] {
        std::vector<RigTemplateInfo> result;
        for(const auto& d:definitions())result.push_back({d.id,d.label,d.family,d.joints.size(),d.defaultHeight,d.version});
        return result;
    }();return catalogue;
}
bool buildRigTemplate(const std::string& id,const std::string& character,float height,
                      RayTrophi::NodeHierarchy& hierarchy,RigAnatomy& anatomy,std::string& error) {
    error.clear();
    if(!std::isfinite(height) || height<=0 || height>10000){error="invalid_rig_height";return false;}
    if(character.empty() || character.size()>128){error="invalid_rig_name";return false;}
    for(unsigned char c:character)if(!((c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='_'||c=='-')){error="invalid_rig_name";return false;}
    const Definition* definition=nullptr;for(const auto& d:definitions())if(d.id==id){definition=&d;break;}
    if(!definition){error="unknown_rig_template";return false;}
    RayTrophi::NodeHierarchy h;RigAnatomy a=definition->anatomy;a.family=definition->family;
    const auto prefix=character+"_";
    for(size_t i=0;i<definition->joints.size();++i) {
        const auto& joint=definition->joints[i];
        if(joint.parent>=static_cast<int>(i) || (i>0 && joint.parent<0)){error="invalid_rig_template_definition";return false;}
        Vec3 local=joint.position;
        if(joint.parent>=0)local=local-definition->joints[joint.parent].position;
        h.addNode(joint.name,prefix+joint.name,Matrix4x4::translation(local*height),joint.parent);
        if(!joint.role.empty())a.roles.push_back({joint.role,prefix+joint.name});
    }
    for(auto& pair:a.symmetry){pair.left=prefix+pair.left;pair.right=prefix+pair.right;}
    for(auto& chain:a.chains)for(auto& bone:chain.bones)bone=prefix+bone;
    for(auto& rule:a.fitRules) {
        rule.bone=prefix+rule.bone;rule.start=prefix+rule.start;rule.end=prefix+rule.end;
    }
    for (auto& control : a.drivenControls) {
        control.anchor = prefix + control.anchor;
        for (auto& driver : control.drivers)
            driver.bone = prefix + driver.bone;
    }
    if(!validateRigAnatomy(a,h,error))return false;
    hierarchy=std::move(h);anatomy=std::move(a);return true;
}
}
