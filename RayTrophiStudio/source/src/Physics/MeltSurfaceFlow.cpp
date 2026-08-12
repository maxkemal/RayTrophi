#include "MeltSurfaceFlow.h"

#include <algorithm>
#include <cmath>

namespace RayTrophiSim {
namespace {

// Triangle-connectivity adjacency in CSR form.
//
// Deduplicated per vertex on purpose: an interior edge is shared by two
// triangles and would otherwise be listed twice, giving it double the transport
// weight of a boundary edge. That asymmetry would push liquid away from the
// silhouette of an open mesh for no physical reason.
struct Adjacency {
    std::vector<uint32_t> offset;    // size count + 1
    std::vector<uint32_t> neighbour;

    static Adjacency build(std::size_t count,
                           const std::vector<uint32_t>& triangles) {
        Adjacency out;
        std::vector<std::vector<uint32_t>> lists(count);
        auto link = [&](uint32_t a, uint32_t b) {
            if (a >= count || b >= count || a == b) return;
            lists[a].push_back(b);
        };
        for (std::size_t i = 0; i + 2 < triangles.size(); i += 3) {
            const uint32_t a = triangles[i], b = triangles[i + 1], c = triangles[i + 2];
            link(a, b); link(b, a);
            link(b, c); link(c, b);
            link(c, a); link(a, c);
        }
        out.offset.resize(count + 1u, 0u);
        std::size_t total = 0;
        for (std::size_t i = 0; i < count; ++i) {
            auto& list = lists[i];
            std::sort(list.begin(), list.end());
            list.erase(std::unique(list.begin(), list.end()), list.end());
            total += list.size();
        }
        out.neighbour.reserve(total);
        for (std::size_t i = 0; i < count; ++i) {
            out.offset[i] = static_cast<uint32_t>(out.neighbour.size());
            out.neighbour.insert(out.neighbour.end(), lists[i].begin(), lists[i].end());
        }
        out.offset[count] = static_cast<uint32_t>(out.neighbour.size());
        return out;
    }
};

} // namespace

bool solveMeltSurfaceFlow(const std::vector<Vec3>& rest,
                          const std::vector<uint32_t>& triangles,
                          const std::vector<float>& melt_in,
                          const std::vector<float>& local_mass_in,
                          const std::vector<uint8_t>& sampled,
                          const MeltSurfaceFlowSettings& settings,
                          std::vector<Vec3>& out) {
    const std::size_t count = rest.size();
    if (count < 3 || melt_in.size() != count || local_mass_in.size() != count ||
        triangles.size() < 3) return false;

    const Adjacency adjacency = Adjacency::build(count, triangles);

    // ── Fill the vertices the field could not answer for ──────────────────────
    // ★ These are why a melting surface grew spikes. A vertex whose UV lands on
    // no island arrives here with melt = 0 and mass = 1 — the rest-pose values —
    // so while every neighbour around it sank it stayed exactly where it was and
    // became a needle. The defaults were never a reading; they were the absence
    // of one, and geometry cannot tell the difference.
    //
    // The fill spreads through mesh connectivity, not through UV space, because
    // the claim being made is about the SURFACE being continuous. Vertices with
    // no sampled neighbour at any distance keep the defaults and therefore keep
    // their rest position, which is what "nothing is known here" should look
    // like.
    std::vector<float> melt = melt_in;
    std::vector<float> local_mass = local_mass_in;
    if (sampled.size() == count) {
        std::vector<uint8_t> known = sampled;
        std::vector<uint8_t> next;
        // Unmapped runs are thin (a sliver between islands); a handful of passes
        // crosses them. An unbounded flood would spend its time on the genuinely
        // unmapped components that are meant to stay put.
        for (int pass = 0; pass < 6; ++pass) {
            next = known;
            bool filled_any = false;
            for (std::size_t i = 0; i < count; ++i) {
                if (known[i]) continue;
                float sum_melt = 0.0f, sum_mass = 0.0f;
                uint32_t n_known = 0u;
                for (uint32_t n = adjacency.offset[i]; n < adjacency.offset[i + 1u]; ++n) {
                    const uint32_t j = adjacency.neighbour[n];
                    if (!known[j]) continue;
                    sum_melt += melt[j];
                    sum_mass += local_mass[j];
                    ++n_known;
                }
                if (n_known == 0u) continue;
                melt[i] = sum_melt / static_cast<float>(n_known);
                local_mass[i] = sum_mass / static_cast<float>(n_known);
                next[i] = 1u;
                filled_any = true;
            }
            known.swap(next);
            if (!filled_any) break;
        }
    }

    // ── Per-vertex surface area: a third of each incident triangle ────────────
    // Barycentric-dual area. This is what turns a volume into a thickness, so it
    // has to come from the REST pose (the deformed pose would feed the solve its
    // own output and let a thinning vertex thin itself further every solve).
    std::vector<float> area(count, 0.0f);
    for (std::size_t i = 0; i + 2 < triangles.size(); i += 3) {
        const uint32_t a = triangles[i], b = triangles[i + 1], c = triangles[i + 2];
        if (a >= count || b >= count || c >= count) continue;
        const float tri_area =
            Vec3::cross(rest[b] - rest[a], rest[c] - rest[a]).length() * 0.5f;
        if (!(tri_area > 0.0f)) continue;
        area[a] += tri_area / 3.0f;
        area[b] += tri_area / 3.0f;
        area[c] += tri_area / 3.0f;
    }

    // ── Split each vertex's remaining material into immobile and mobile ───────
    // remaining = what is still attached here at all (APIC transfer and
    //             pyrolysis have already been subtracted upstream)
    // molten    = the part of it that is liquid, and therefore transportable
    //
    // `melt` is the GEOMETRIC melt history — local liquid plus what has already
    // been handed to APIC — so min() against `remaining` is doing real work, not
    // defensive clamping. Treating the survivor of a transfer as liquid is the
    // physically right reading rather than a concession: a patch that has been
    // shedding molten mass is by definition sitting at its melt point, so what
    // is left of it is liquid too. The volume is exact either way, because the
    // two fractions are split out of `remaining` and always sum back to it.
    std::vector<float> solid_fraction(count, 0.0f);
    std::vector<float> volume(count, 0.0f);          // mobile, fraction * m^2
    double total_mobile = 0.0;
    bool any_material_missing = false;
    for (std::size_t i = 0; i < count; ++i) {
        const float remaining = std::clamp(local_mass[i], 0.0f, 1.0f);
        const float molten = std::clamp(melt[i], 0.0f, 1.0f);
        const float mobile = std::min(molten, remaining);
        solid_fraction[i] = remaining - mobile;
        volume[i] = mobile * area[i];
        total_mobile += volume[i];
        if (remaining < 1.0f) any_material_missing = true;
    }
    // Nothing liquid and nothing gone: the rest pose IS the answer, and saying so
    // here keeps the common case at the cost of one scan.
    if (total_mobile <= 0.0 && !any_material_missing) { out = rest; return true; }

    float min_y = rest[0].y, max_y = rest[0].y;
    for (std::size_t i = 0; i < count; ++i) {
        min_y = std::min(min_y, rest[i].y);
        max_y = std::max(max_y, rest[i].y);
    }
    const float object_height = std::max(max_y - min_y, 1.0e-4f);
    const float height_scale =
        object_height * std::clamp(settings.maximum_height_loss, 0.0f, 1.0f);

    // ── Downhill transport ────────────────────────────────────────────────────
    // Jacobi sweeps: every vertex offers a fixed share of its liquid to its
    // downhill neighbours, split by how far downhill each one is. What leaves a
    // vertex arrives at another, so the sum is conserved to floating point — the
    // whole reason this is written as transport and not as a diffusion of the
    // melt value.
    //
    // The height that drives the flow is the CURRENT surface, liquid included.
    // Using the rest height instead would let a puddle keep receiving material
    // after it had already piled into a spike, because the destination would
    // look permanently low.
    //
    // ★★ THE OTHER SPIKE SOURCE, and the one that looked like boiling. This used
    // to offer a flat `volume[i] * flow_rate` split by height difference alone.
    // Two things went wrong, and both are fixed by sizing the offer in the units
    // the constraint is actually expressed in:
    //
    //   - Nothing bounded the offer by how far downhill the neighbour was, so a
    //     vertex barely above its neighbour still handed over half its liquid,
    //     dropped below it, and got it back on the next sweep. A Jacobi sweep
    //     that overshoots oscillates, and since the solve reruns on every mask
    //     revision the surface flipped between phases of that oscillation from
    //     frame to frame. That is the "rapid boiling".
    //   - A share sized in VOLUME becomes height by dividing by the receiver's
    //     own area, so a vertex in a dense patch turned an ordinary share into a
    //     needle. Weighting the shares by area removes the division entirely: a
    //     receiver's height gain no longer depends on how finely it is tessellated.
    //
    // `level` is the volume that would put i and j at exactly the same height
    // (moving V drops i by V/area_i and raises j by V/area_j). Capping each share
    // at a fraction of its own `level` makes overshoot impossible by construction
    // rather than by tuning, so the sweep is monotone and settles instead of
    // ringing.
    const float flow_rate =
        std::clamp(1.0f - std::clamp(settings.viscosity, 0.0f, 1.0f), 0.0f, 1.0f) * 0.5f;
    std::vector<float> surface_y(count, 0.0f);
    std::vector<float> delta(count, 0.0f);
    std::vector<float> offer(adjacency.neighbour.size(), 0.0f);
    if (flow_rate > 0.0f && total_mobile > 0.0) {
        for (uint32_t pass = 0; pass < settings.flow_iterations; ++pass) {
            for (std::size_t i = 0; i < count; ++i) {
                // An area-less vertex belongs to no non-degenerate triangle, so
                // there is no thickness to speak of: it reads as rest height and
                // takes no part in transport.
                const float thickness = area[i] > 0.0f
                    ? solid_fraction[i] + volume[i] / area[i] : 1.0f;
                surface_y[i] = rest[i].y + (thickness - 1.0f) * height_scale;
            }
            std::fill(delta.begin(), delta.end(), 0.0f);
            bool moved = false;
            for (std::size_t i = 0; i < count; ++i) {
                if (!(volume[i] > 0.0f) || !(area[i] > 0.0f)) continue;
                const uint32_t begin = adjacency.offset[i];
                const uint32_t end = adjacency.offset[i + 1u];
                float offer_total = 0.0f;
                float drop_max = 0.0f;
                for (uint32_t n = begin; n < end; ++n) {
                    const uint32_t j = adjacency.neighbour[n];
                    offer[n] = 0.0f;
                    if (!(area[j] > 0.0f)) continue;
                    const float d = surface_y[i] - surface_y[j];
                    if (!(d > 0.0f)) continue;
                    offer[n] = d * (area[i] * area[j]) / (area[i] + area[j]);
                    offer_total += offer[n];
                    drop_max = std::max(drop_max, d);
                }
                if (!(offer_total > 0.0f)) continue;
                // Three limits, all binding at once: what is here, what levelling
                // asks for, and how far this vertex may sink in one sweep.
                const float send = std::min(volume[i],
                    std::min(offer_total, drop_max * area[i]) * flow_rate);
                if (!(send > 0.0f)) continue;
                const float norm = send / offer_total;
                for (uint32_t n = begin; n < end; ++n) {
                    if (!(offer[n] > 0.0f)) continue;
                    const float share = offer[n] * norm;
                    delta[adjacency.neighbour[n]] += share;
                    delta[i] -= share;
                }
                moved = true;
            }
            if (!moved) break;  // everything is already level
            for (std::size_t i = 0; i < count; ++i)
                volume[i] = std::max(volume[i] + delta[i], 0.0f);
        }
    }

    // ── Resolve to geometry ───────────────────────────────────────────────────
    // thickness == 1 is the rest surface. Below 1 the vertex has lost material
    // (melted away and flowed off, burnt, or handed to APIC) and sinks; above 1
    // liquid has pooled here and it rises.
    std::vector<float> thickness(count, 1.0f);
    Vec3 pool_axis(0.0f, 0.0f, 0.0f);
    float pool_weight = 0.0f;
    for (std::size_t i = 0; i < count; ++i) {
        thickness[i] = area[i] > 0.0f
            ? solid_fraction[i] + volume[i] / area[i] : 1.0f;
        const float gain = std::max(thickness[i] - 1.0f, 0.0f);
        if (gain > 0.0f) { pool_axis += rest[i] * gain; pool_weight += gain; }
    }
    if (pool_weight > 1.0e-6f) pool_axis *= 1.0f / pool_weight;

    const float lateral_cap = std::max(settings.maximum_lateral_gain, 0.0f);
    out = rest;
    for (std::size_t i = 0; i < count; ++i) {
        const float offset = (thickness[i] - 1.0f) * height_scale;
        // Never push a vertex through the object's own rest floor: without
        // contact against the surface it is standing on, that is the closest
        // stand-in for "the puddle has reached the ground".
        out[i].y = std::max(rest[i].y + offset, min_y);

        if (pool_weight <= 1.0e-6f) continue;
        const float gain = std::max(thickness[i] - 1.0f, 0.0f);
        if (!(gain > 0.0f)) continue;
        // Approximate lateral spread: liquid that piled up here also widens.
        Vec3 outward(rest[i].x - pool_axis.x, 0.0f, rest[i].z - pool_axis.z);
        const float radius = std::sqrt(outward.x * outward.x + outward.z * outward.z);
        if (!(radius > 1.0e-5f)) continue;
        const float push = std::min(gain, lateral_cap) * height_scale;
        out[i].x += (outward.x / radius) * push;
        out[i].z += (outward.z / radius) * push;
    }
    return true;
}

} // namespace RayTrophiSim
