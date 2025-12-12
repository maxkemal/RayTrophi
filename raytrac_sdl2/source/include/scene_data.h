#pragma once
#include <HittableList.h>
#include <AssimpLoader.h>
#include <AnimatedObject.h>
struct SceneData {
    HittableList world;
    std::shared_ptr<Hittable> bvh;
    std::vector<AnimationData> animationDataList;
    std::vector<std::shared_ptr<AnimatedObject>> animatedObjects;
    std::shared_ptr<Camera> camera;
    std::vector<std::shared_ptr<Light>> lights;
    Vec3 background_color=Vec3(0.1,0.2,0.3);
    bool initialized = false;
    BoneData boneData;
    ColorProcessor color_processor;
    void clear() {
        world.clear();
        lights.clear();
        animatedObjects.clear();
        animationDataList.clear();
        camera = nullptr;
        bvh = nullptr;
        initialized = false;
    }

};