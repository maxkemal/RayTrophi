#pragma once
#include "Animation/RigIK.h"
#include <map>
namespace RigAuthoring {
struct IKControlKey {double seconds=0;IKPose pose;};
struct IKContactInterval {double start=0,end=0;IKPose pose;};
struct IKChannel {std::vector<IKControlKey> keys;std::vector<IKContactInterval> contacts;};
using IKChannels=std::map<std::string,IKChannel>;
bool validateIKChannels(const IKChannels&,std::string& error);
bool validateIKChannelControls(const IKChannels&,const std::vector<IKControl>&,std::string& error);
IKPoses sampleIKChannels(const IKChannels&,double localSeconds);
bool insertIKChannelKey(IKChannels&,const std::string& control,double seconds,const IKPose&,std::string& error);
bool removeIKChannelKey(IKChannels&, const std::string& control, double seconds,
                        std::string& error);
nlohmann::json serializeIKChannels(const IKChannels&);
bool deserializeIKChannels(const nlohmann::json&,IKChannels&,std::string& error);
}
