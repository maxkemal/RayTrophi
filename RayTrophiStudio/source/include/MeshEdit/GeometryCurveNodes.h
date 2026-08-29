#pragma once

// Included by GeometryNodesV2.h after GeometryNodeBase and NodeType are defined.

class SplineObjectNode final : public GeometryNodeBase {
public:
    char objectName[128] = "";

    SplineObjectNode();
    std::string getTypeId() const override;
    void serializeParams(nlohmann::json& j) const override;
    void deserializeParams(const nlohmann::json& j) override;
    void drawContent() override;
    NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
};

class ResampleCurveNode final : public GeometryNodeBase {
public:
    int samples = 32;
    bool uniformDistance = true;

    ResampleCurveNode();
    std::string getTypeId() const override;
    void serializeParams(nlohmann::json& j) const override;
    void deserializeParams(const nlohmann::json& j) override;
    void drawContent() override;
    NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
};

class CurveTaperNode final : public GeometryNodeBase {
public:
    float startScale = 1.0f;
    float endScale = 0.0f;
    float exponent = 1.0f;

    CurveTaperNode();
    std::string getTypeId() const override;
    void serializeParams(nlohmann::json& j) const override;
    void deserializeParams(const nlohmann::json& j) override;
    void drawContent() override;
    NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
};

class CurveTwistNode final : public GeometryNodeBase {
public:
    float startDegrees = 0.0f;
    float endDegrees = 360.0f;

    CurveTwistNode();
    std::string getTypeId() const override;
    void serializeParams(nlohmann::json& j) const override;
    void deserializeParams(const nlohmann::json& j) override;
    void drawContent() override;
    NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
};

class CurveWaveNode final : public GeometryNodeBase {
public:
    float amplitude = 0.1f;
    float frequency = 1.0f;
    float phaseDegrees = 0.0f;
    float noise = 0.0f;
    int seed = 0;
    int axis = 1;

    CurveWaveNode();
    std::string getTypeId() const override;
    void serializeParams(nlohmann::json& j) const override;
    void deserializeParams(const nlohmann::json& j) override;
    void drawContent() override;
    NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
};

class CurveToMeshNode final : public GeometryNodeBase {
public:
    int pathSamples = 48;
    int profileSamples = 12;
    float radius = 0.1f;
    bool usePointRadius = true;
    bool capStart = true;
    bool capEnd = true;
    int materialId = -1; // -1 inherits the host mesh's first material slot.

    CurveToMeshNode();
    std::string getTypeId() const override;
    void serializeParams(nlohmann::json& j) const override;
    void deserializeParams(const nlohmann::json& j) override;
    void drawContent() override;
    NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
};

class GeometryNodeGraphV2;
std::size_t curveGraphSourceSignature(
    const GeometryNodeGraphV2& graph,
    const std::vector<std::shared_ptr<Hittable>>& sceneObjects,
    bool* usesCurves = nullptr);
