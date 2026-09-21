#include "RayFusion/ProbeField.h"

#include <cmath>
#include <limits>
#include <set>

namespace RayFusion {
std::vector<Check> validateProbeCore() {
    std::vector<Check> checks;
    auto check = [&](const char* name, bool passed) {
        checks.push_back({name, passed, passed ? "passed" : "control-plane invariant failed"});
    };
    auto near = [](float a, float b) { return std::abs(a - b) < 1e-5f; };
    auto packet = [](float value) {
        ProbePacket data{};
        for (auto& texel : data) {
            texel.irradiance = {value, value * 0.5f, value * 0.25f, 0};
            texel.distance = {2, 4, 0, 0};
        }
        return data;
    };
    const Revision revision{1, 1, 1, 1};
    const Budget all{32, 64, 2048};
    const auto light = packet(4);
    std::string error;
    Grid grid;
    grid.counts = {2, 2, 2};
    grid.minimum = {-1, -1, -1};
    grid.targetUpdates = 1;
    ProbeField field;
    check("unconfigured_has_no_work", field.schedule(all).empty() && field.stats().total == 0);
    const bool configured = field.configure(grid, revision, error);
    check("configure_unknown_is_not_black_measurement", configured && field.stats().valid == 0 &&
        field.stats().pending == 8 && field.lookup({0, 0, 0}) == nullptr);
    if (!configured) return checks;

    check("sub_probe_ray_budget_does_not_dispatch", field.schedule({32, 4, 31}).empty());
    auto batch = field.schedule({32, 100, 65});
    check("integer_ray_budget_respected", batch.size() == 2);
    std::set<uint32_t> slots;
    for (const auto& ticket : batch) slots.insert(ticket.slot);
    auto second = field.schedule(all);
    for (const auto& ticket : second) slots.insert(ticket.slot);
    check("inflight_not_scheduled_twice", slots.size() == 8 && second.size() == 6 && field.schedule(all).empty());
    batch.insert(batch.end(), second.begin(), second.end());
    if (batch.size() != 8) return checks;

    bool allAccepted = true;
    for (const auto& ticket : batch) allAccepted &= field.publish(ticket, light, 0.98f, error);
    const auto* first = field.lookup(batch[0].cell);
    check("first_sample_not_darkened_by_empty_history", allAccepted && first && near((*first)[0].irradiance[0], 4));
    check("complete_field_stops_scheduling", field.stats().valid == 8 && field.stats().pending == 0 && field.schedule(all).empty());
    check("duplicate_completion_rejected", !field.publish(batch[0], light, 0, error));
    check("same_revision_retains_lighting", field.invalidate(revision, error) && field.stats().valid == 8);

    check("scroll_retains_overlapping_world_cells", field.scroll({0, -1, -1}, error) &&
        field.stats().valid == 4 && field.lookup({0, 0, 0}) != nullptr && field.lookup({1, 0, 0}) == nullptr);
    auto recycled = field.schedule(all);
    check("scroll_dispatches_only_new_cells", recycled.size() == 4);
    field.scroll({-1, -1, -1}, error);
    check("recycled_slot_rejects_old_world_result", !recycled.empty() && !field.publish(recycled[0], light, 0, error));
    field.scroll({100, 100, 100}, error);
    check("large_camera_jump_has_no_old_irradiance", field.stats().valid == 0 && field.stats().pending == 8);

    auto beforeEpoch = field.schedule(all);
    Revision next = revision;
    ++next.sceneEpoch;
    field.invalidate(next, error);
    check("old_scene_completion_rejected", !beforeEpoch.empty() && !field.publish(beforeEpoch[0], light, 0, error));
    auto beforeDevice = field.schedule(all);
    ++next.deviceEpoch;
    field.invalidate(next, error);
    check("old_device_completion_rejected", !beforeDevice.empty() && !field.publish(beforeDevice[0], light, 0, error));
    auto beforeLight = field.schedule(all);
    ++next.lighting;
    field.invalidate(next, error);
    check("old_light_completion_rejected", !beforeLight.empty() && !field.publish(beforeLight[0], light, 0, error));
    auto beforeGeometry = field.schedule(all);
    ++next.geometry;
    field.invalidate(next, error);
    check("old_geometry_completion_rejected", !beforeGeometry.empty() && !field.publish(beforeGeometry[0], light, 0, error));

    auto active = field.schedule({32, 1, 32});
    check("invalidated_field_accepts_new_work", active.size() == 1);
    if (active.empty()) return checks;
    ProbePacket invalid = light;
    invalid.back().irradiance[2] = (std::numeric_limits<float>::quiet_NaN)();
    check("invalid_packet_is_atomic_and_retryable", !field.publish(active[0], invalid, 0, error) &&
        field.stats().valid == 0 && field.stats().inFlight == 1);
    invalid = light;
    invalid.back().distance[1] = 1;
    check("impossible_distance_moments_rejected", !field.publish(active[0], invalid, 0, error));
    check("invalid_history_rejected", !field.publish(active[0], light, 1, error));
    check("cancel_releases_failed_work", field.cancel(active[0]) && field.stats().inFlight == 0);
    check("cancelled_completion_rejected", !field.publish(active[0], light, 0, error));

    Grid bad = grid;
    bad.counts = {128, 128, 128};
    check("oversize_grid_rejected_without_mutation", !field.configure(bad, revision, error) && field.stats().total == 8);
    bad = grid;
    bad.spacing = (std::numeric_limits<float>::infinity)();
    check("nonfinite_grid_rejected", !field.configure(bad, revision, error));
    check("unowned_grid_rejected", !field.configure(grid, {}, error));

    ProbeField a, b;
    grid.counts = {1, 1, 1};
    grid.minimum = {0, 0, 0};
    grid.targetUpdates = 2;
    a.configure(grid, revision, error);
    b.configure(grid, revision, error);
    auto ta = a.schedule(all);
    auto tb = b.schedule(all);
    check("cross_field_completion_rejected", !ta.empty() && !tb.empty() && !b.publish(ta[0], light, 0, error));
    if (!ta.empty()) {
        a.publish(ta[0], packet(2), 0.98f, error);
        auto update = a.schedule(all);
        const bool blended = !update.empty() && a.publish(update[0], packet(4), 0.5f, error);
        const auto* result = a.lookup({0, 0, 0});
        check("same_epoch_temporal_blend", blended && result && near((*result)[0].irradiance[0], 3));
        a.configure(grid, revision, error);
        check("reconfigure_rejects_previous_tickets", !a.publish(ta[0], light, 0, error));
    }

    ProbeField retained;
    grid.counts = {2, 1, 1};
    retained.configure(grid, revision, error);
    auto moving = retained.schedule(all);
    retained.scroll({1, 0, 0}, error);
    bool kept = false, discarded = false;
    for (const auto& ticket : moving) {
        const bool accepted = retained.publish(ticket, light, 0, error);
        if (ticket.cell[0] == 1) kept = accepted;
        if (ticket.cell[0] == 0) discarded = !accepted;
    }
    check("scroll_preserves_retained_inflight_work", kept && discarded);
    check("planar_blocker_front_visible", near(momentVisibility(2, 4, 1), 1));
    check("planar_blocker_back_occluded", near(momentVisibility(2, 4, 3), 0));
    check("moment_visibility_bounded", near(momentVisibility(2, 5, 3), 0.125f));
    check("invalid_visibility_is_not_unoccluded", near(momentVisibility(2, 1, 1), 0));
    return checks;
}
} // namespace RayFusion
