#pragma once

#include "SceneCommand.h"

#include <memory>
#include <string>

namespace DNA { class GeometryDetail; }

class FlatMeshGeometryCommand final : public SceneCommand {
public:
    FlatMeshGeometryCommand(std::string object_name,
                            std::shared_ptr<DNA::GeometryDetail> before,
                            std::shared_ptr<DNA::GeometryDetail> after,
                            std::string description);

    void execute(UIContext& ctx) override;
    void undo(UIContext& ctx) override;
    Type getType() const override { return Type::Heavy; }
    std::string getDescription() const override { return description_; }
    bool isHeavyGeometry() const override { return true; }
    size_t getTriangleCount() const override;

private:
    std::string object_name_;
    std::shared_ptr<DNA::GeometryDetail> before_;
    std::shared_ptr<DNA::GeometryDetail> after_;
    std::string description_;

    void apply(UIContext& ctx, const std::shared_ptr<DNA::GeometryDetail>& geometry);
};
