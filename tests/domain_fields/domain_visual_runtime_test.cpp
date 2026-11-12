#include "Fluid/DomainFieldStorage.h"
#include "Fluid/DomainVisualRuntime.h"

#include <cstdlib>
#include <iostream>
#include <utility>

using namespace RayTrophiSim::Fluid::DomainFields;

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

template<class Operation>
void rejects(Operation operation, const char* message) {
    bool rejected = false;
    try {
        operation();
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, message);
}

std::vector<ChannelBuffer> uniformBuffers() {
    std::vector<ChannelBuffer> buffers;
    constexpr Location locations[] = {Location::FaceX, Location::FaceY, Location::FaceZ};
    for (int axis = 0; axis < 3; ++axis) {
        ChannelBuffer channel;
        channel.channel = static_cast<Channel>(axis);
        channel.description = {locations[axis], Unit::LengthPerSecond,
                               Residence::Host, 0.0, true};
        channel.first = {-1, -1, -1};
        channel.extent = {7, 7, 7};
        channel.values.assign(343, axis == 0 ? 0.1f : 0.0f);
        buffers.push_back(std::move(channel));
    }
    return buffers;
}

Identity identity(std::uint64_t generation, double time) {
    Identity id;
    id.domain_uid = "test-domain";
    id.state_generation = generation;
    id.step = generation;
    id.physical_time = time;
    return id;
}

} // namespace

int main() {
    const Geometry geometry{{-2.0, -2.0, -2.0}, {4, 4, 4}, 1.0};
    auto buffers = uniformBuffers();
    const auto frozen = FrozenSnapshot::publish(identity(1, 0.0), geometry, buffers, 10000);
    buffers[0].values.assign(343, 900.0f);
    require(std::abs(sampleMacVelocity(*frozen, {-0.5, -0.5, -0.5}).value[0] - 0.1) < 1e-7,
            "producer mutation changed immutable publication");
    rejects([&] { FrozenSnapshot::publish(identity(1, 0.0), geometry, buffers, 1); },
            "CPU byte budget ignored");
    buffers = uniformBuffers();
    buffers.push_back(buffers[0]);
    rejects([&] { FrozenSnapshot::publish(identity(1, 0.0), geometry, buffers, 10000); },
            "duplicate channel accepted");
    buffers = uniformBuffers();
    buffers[0].values[0] = std::numeric_limits<float>::infinity();
    rejects([&] { FrozenSnapshot::publish(identity(1, 0.0), geometry, buffers, 10000); },
            "non-finite host field accepted");

    VisualSampleSettings settings;
    settings.target_volume = 0.125;
    settings.max_samples = 32;
    VisualRuntime runtime;
    runtime.reset(settings);
    const std::vector<SampleRegion> regions{{{1, 1, 1}, 7, 1.0}};
    const VisualSupport support = [](std::uint64_t, const std::array<double, 3>&) {
        return MatterSupport{true, Validity::ValidData};
    };
    const TraceVisualSegment trace = [](std::uint64_t, const std::array<double, 3>&,
                                        std::array<double, 3>&, double, double) {
        return Validity::ValidData;
    };
    const auto seeded = runtime.update(*frozen, regions, trace, support);
    require(seeded.reseed.born == 8 && runtime.state().samples.size() == 8,
            "volume-based sample allocation incorrect");
    const auto initial = runtime.state();
    require(runtime.update(*frozen, regions, trace, support).unchanged,
            "redraw advanced visual state");
    require(runtime.state().next_uid == initial.next_uid, "redraw consumed identities");

    const auto next = FrozenSnapshot::publish(identity(2, 0.1), geometry, uniformBuffers(), 10000);
    runtime.update(*next, regions, trace, support);
    std::size_t retained = 0;
    for (const auto& sample : runtime.state().samples) {
        const auto found = std::find_if(initial.samples.begin(), initial.samples.end(),
            [&](const auto& old) { return old.uid == sample.uid; });
        if (found != initial.samples.end()) {
            ++retained;
            require(std::abs(sample.position[0] - found->position[0] - 0.01) < 1e-7,
                    "uniform velocity transport differs from analytic trajectory");
        }
    }
    require(retained > 0, "transport oracle retained no samples");
    const auto at_next = runtime.state();
    const auto later = FrozenSnapshot::publish(identity(3, 0.2), geometry, uniformBuffers(), 10000);
    const TraceVisualSegment unavailable = [](std::uint64_t,
        const std::array<double, 3>&, std::array<double, 3>&, double, double) {
        return Validity::Unavailable;
    };
    rejects([&] { runtime.update(*later, regions, unavailable, support); },
            "unavailable transport published partial-time samples");
    require(runtime.state().identity.state_generation == at_next.identity.state_generation &&
            runtime.state().next_uid == at_next.next_uid, "failed transport was not atomic");
    rejects([&] { runtime.update(*frozen, regions, trace, support); },
            "rewind silently discarded sample history");
    runtime.restore(initial);
    runtime.update(*next, regions, trace, support);
    require(runtime.state().samples.size() == at_next.samples.size(),
            "checkpoint replay count differs");
    for (std::size_t i = 0; i < at_next.samples.size(); ++i) {
        require(runtime.state().samples[i].uid == at_next.samples[i].uid &&
                runtime.state().samples[i].position == at_next.samples[i].position,
                "checkpoint replay changed trajectory or identity");
    }
    auto invalid_checkpoint = initial;
    invalid_checkpoint.samples[0].uid = invalid_checkpoint.next_uid;
    rejects([&] { runtime.restore(invalid_checkpoint); }, "invalid checkpoint accepted");

    std::vector<VisualSample> a;
    std::vector<VisualSample> b;
    std::uint64_t uid_a = 1;
    std::uint64_t uid_b = 1;
    const std::vector<SampleRegion> two{{{1, 1, 1}, 7, 1.0}, {{2, 1, 1}, 8, 0.5}};
    auto reversed = two;
    std::reverse(reversed.begin(), reversed.end());
    settings.max_samples = 3;
    const auto report = reseedVisualSamples(geometry, settings, 0.0, two, support, uid_a, a);
    reseedVisualSamples(geometry, settings, 0.0, reversed, support, uid_b, b);
    require(a.size() == 3 && b.size() == 3 && uid_a == uid_b,
            "sample budget or permutation invariant failed");
    double volume = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        volume += a[i].represented_volume;
        require(a[i].uid == b[i].uid && a[i].position == b[i].position &&
                a[i].stream_uid == b[i].stream_uid, "region order changed visual samples");
    }
    require(std::abs(volume - 1.5) < 1e-12 && report.unrepresented_volume == 0.0,
            "budget reduction changed represented physical volume");
    const auto no_support = [](std::uint64_t, const std::array<double, 3>&) {
        return MatterSupport{false, Validity::Unavailable};
    };
    rejects([&] { reseedVisualSamples(geometry, settings, 0.0, two, no_support, uid_a, a); },
            "unavailable support retired live samples");
    const auto known_empty = [](std::uint64_t, const std::array<double, 3>&) {
        return MatterSupport{false, Validity::KnownBackground};
    };
    const auto empty = reseedVisualSamples(geometry, settings, 0.0, two, known_empty, uid_a, a);
    require(a.empty() && empty.unrepresented_volume == 1.5,
            "unavailable support manufactured geometry");

    settings.max_samples = 32;
    a = initial.samples;
    uid_a = initial.next_uid;
    const std::vector<SampleRegion> slight_change{{{1, 1, 1}, 7, 1.05}};
    const auto stable = reseedVisualSamples(geometry, settings, 0.0, slight_change,
                                           support, uid_a, a);
    require(stable.born == 0 && stable.retired == 0 && a.size() == initial.samples.size(),
            "reseed hysteresis replaced a stable population");
    const auto before_failure = a;
    const auto before_uid = uid_a;
    auto duplicate_regions = two;
    duplicate_regions.push_back(two[0]);
    rejects([&] { reseedVisualSamples(geometry, settings, 0.0, duplicate_regions,
                                    support, uid_a, a); }, "duplicate support region accepted");
    require(uid_a == before_uid && a[0].position == before_failure[0].position,
            "failed reseed was not atomic");

    settings.max_substeps = 1;
    settings.cfl = 0.001;
    a = initial.samples;
    const auto held = transportVisualSamples(*next, settings, 0.0, 0.1, trace, a);
    require(held.held == a.size() && a[0].position == initial.samples[0].position,
            "CFL exhaustion left partially advanced samples");
    runtime.restore(initial);
    settings.max_substeps = 256;
    settings.cfl = 0.5;
    settings.max_samples = 2;
    runtime.configure(settings);
    runtime.update(*frozen, regions, trace, support, true);
    require(runtime.state().samples.size() == 2,
            "same-publication budget edit did not reseed to the new quota");
    std::cout << "domain visual runtime PASS\n";
}
