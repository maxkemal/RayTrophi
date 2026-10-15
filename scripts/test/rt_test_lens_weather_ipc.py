"""Post-build L1 probe, external terminal only; Studio must already be open.

Default: settings, atomic rejection and actual GPU resource status. --wet also
temporarily aims the camera upward and advances 30 timeline frames in an already
wet rain/drizzle scene with the global timeline paused, then checks frozen/replayed drop state. Camera position,
target, timeline and settings are restored. Transient glass history is cleared;
scene simulation side effects of seeking are not rolled back. No files created.
--snow uses the same setup with falling snow at >=1 mm/h and temporarily changes
the common climate temperature to test melt/refreeze (2% total-water budget).
The original climate temperature is restored; automatic cloud derivation follows
these edits. Run only in a saved test scene, not during an animated climate track.
"""
import argparse
import time
from rt_ipc import RtIpc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("viewport", "render"), default="viewport")
    parser.add_argument("--wet", action="store_true")
    parser.add_argument("--snow", action="store_true",
                        help="Snow accumulation/replay/melt; paused saved snow scene required")
    args = parser.parse_args()
    if args.wet and args.snow:
        parser.error("Choose --wet or --snow")
    client = RtIpc()
    original = client.call("camera.get_lens_weather")
    camera = client.call("camera.get")
    frame = client.call("timeline.get_frame")
    moved = False
    original_climate = client.call("world.get_climate")
    climate_changed = False

    def set_temperature(celsius):
        nonlocal climate_changed
        local = client.call("world.sample_climate", position=camera["position"])
        sea_kelvin = celsius + 273.15 + original_climate["lapse_rate_k_per_m"] * min(
            local["altitude_m"], 11000)
        client.call("world.set_climate", surface_temperature_k=sea_kelvin)
        climate_changed = True
        local_weather = client.call("world.sample_weather_local",
                                    points=[camera["position"]])["points"][0]
        assert abs(local_weather["air_temperature_c"] - celsius) < 0.01, local_weather

    def sample():
        result = client.call("camera.sample_lens_weather", backend=args.backend)
        assert result["resources_ready"] and result["pipelines_ready"], result
        return result

    def set_frame(value, previous):
        client.call("timeline.set_frame", frame=value)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            result = sample()
            if (result["target_step"] != previous and result["step"] == result["target_step"]
                    and result["initialized"] and result["state_matches_camera"]):
                return result
            time.sleep(0.03)
        raise AssertionError("No completed lens display at requested frame; keep the backend visible")

    try:
        for patch in ({"enabled": 1}, {"coating": "unknown"}, {"hood_deg": 90},
                      {"diameter_mm": 9}, {"seed": -1}, {"seed": 1.5},
                      {"seed": True}, {"glass_temperature_c": 0},
                      {"enabled": True, "unknown": 1}):
            before = client.call("camera.get_lens_weather")
            accepted, error = client.try_call("camera.set_lens_weather", patch=patch)
            assert not accepted, (patch, error)
            assert client.call("camera.get_lens_weather") == before
        configured = client.call("camera.set_lens_weather", patch={
            "enabled": True, "coating": "bare" if args.snow else "hydrophobic", "hood_deg": 0,
            "diameter_mm": 200, "seed": 17,
        })
        assert configured["enabled"] and configured["seed"] == 17
        assert configured["coating"] == ("bare" if args.snow else "hydrophobic")
        assert "glass_temperature_c" not in configured
        if args.snow:
            set_temperature(-5)
        assert configured["diameter_mm"] == 200
        state = sample()
        climate = client.call("world.sample_climate", position=camera["position"])
        assert state["temperature_source"] == "atmosphere", state
        assert abs(state["atmosphere_temperature_c"] - (climate["temperature_k"] - 273.15)) < 0.01
        local = client.call("world.sample_weather_local", points=[camera["position"]])["points"][0]
        assert abs(local["air_temperature_c"] - (climate["temperature_k"] - 273.15)) < 0.01
        for offset in (-500, 0, 500):
            point = list(camera["position"])
            point[1] += offset
            measured = client.call("world.sample_climate", position=point)
            weather = client.call("world.sample_weather_local", points=[point])["points"][0]
            temperature = measured["temperature_k"] - 273.15
            assert abs(weather["air_temperature_c"] - temperature) < 0.01
            thaw = min(max(temperature / 2, 0), 1)
            expected = 1 - thaw * thaw * (3 - 2 * thaw)
            assert abs(weather["frozen_surface_share"] - expected) < 0.0001
        print("PASS: shared atmosphere temperature, altitude and surface thermal suitability")
        assert state["capacity"] == 256 and state["history_steps"] == 601
        print("PASS: settings round-trip, atomic validation, GPU pipelines and resource limits")
        if args.wet or args.snow:
            point = camera["position"]
            near = client.call("world.sample_clouds", mode="near_precip",
                               points=[point], backend=args.backend)["points"][0]
            forms = ("snow",) if args.snow else ("rain", "drizzle")
            assert near["rate_mm_h"] >= 1 and near["dominant_form"] in forms, near
            client.call("camera.set_target", target=[point[0] + 0.01, point[1] + 10, point[2]])
            moved = True
            client.call("camera.reset_lens_weather")
            # Allow the clear/display request to be submitted before moving time.
            time.sleep(0.1)
            previous = sample()["target_step"]
            records = {}
            for offset in range(1, 31):
                state = set_frame(frame + offset, previous)
                previous = state["target_step"]
                records[offset] = state
            assert state["impact_rate_per_s"] > 0, state
            assert state["drop_count"] > 0 and state["water_volume_m3"] > 0, state
            if args.snow:
                assert state["snow_water_volume_m3"] > 0 and state["snow_count"] > 0, state
                assert state["field_max_snow"] > 0, state
            else:
                assert state["field_wet_pixels"] > 0, state
                assert state["field_max_coverage"] > 0, state
                assert state["field_max_shift_uv"] > 0, state
            frozen = sample()
            assert frozen["state_hash"] == state["state_hash"], (frozen, state)
            replay = set_frame(frame + 20, previous)
            assert replay["state_hash"] == records[20]["state_hash"], (replay, records[20])
            assert replay["coverage"] == records[20]["coverage"]
            print("PASS: wet arrivals, positive water volume, pause and recent-history replay")
            if args.snow:
                # Keep the camera stationary and face away from falling snow;
                # new arrivals stop, leaving phase transfer as the measured change.
                client.call("camera.set_target", target=[point[0] + 0.01, point[1] - 10, point[2]])
                set_temperature(20)
                previous = replay["target_step"]
                for offset in range(21, 29):
                    melted = set_frame(frame + offset, previous)
                    previous = melted["target_step"]
                assert melted["impact_rate_per_s"] == 0, melted
                assert melted["snow_water_volume_m3"] < replay["snow_water_volume_m3"], melted
                assert melted["liquid_water_volume_m3"] > replay["liquid_water_volume_m3"], melted
                error = abs(melted["water_volume_m3"] / replay["water_volume_m3"] - 1)
                assert error < 0.02, (error, replay, melted)
                print("PASS: snow optical field, warming melts into liquid; total water within 2%")
                set_temperature(-20)
                for offset in range(29, 37):
                    iced = set_frame(frame + offset, previous)
                    previous = iced["target_step"]
                assert iced["impact_rate_per_s"] == 0, iced
                assert iced["ice_water_volume_m3"] > melted["ice_water_volume_m3"], iced
                assert iced["liquid_water_volume_m3"] < melted["liquid_water_volume_m3"], iced
                assert iced["field_max_ice"] > 0, iced
                assert abs(iced["water_volume_m3"] / melted["water_volume_m3"] - 1) < 0.02
                print("PASS: cold glass freezes liquid and produces an ice optical field")
        print("Manual RT/RayFusion refraction, flow, optics and performance acceptance still required")
    finally:
        if moved:
            target = camera["target"]
            client.call("camera.set_target", target=target)
            client.call("timeline.set_frame", frame=frame)
        client.call("camera.set_lens_weather", patch=original)
        if climate_changed:
            client.call("world.set_climate",
                        surface_temperature_k=original_climate["surface_temperature_k"])
        client.call("camera.reset_lens_weather")
        client.close()


if __name__ == "__main__":
    main()
