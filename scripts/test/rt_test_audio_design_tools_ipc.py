"""Post-build layered production, effect rack, full FFT analysis and history acceptance.

Run externally with Studio open. No device audition. Creates/removes only its own
temporary assets/files. Uses bounded short sounds and restores workspace settings.
"""
import argparse
import math
from pathlib import Path
import struct
import tempfile
import wave

from rt_ipc import RtIpc


def rejected(client, method, **params):
    try:
        client.call(method, **params)
    except RuntimeError:
        return
    raise AssertionError(f"Invalid request accepted: {method} {params}")


def signature(report):
    return report["recipe"], report["processed"]["pcm_hash"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    assert args.output_dir.is_dir()
    client = RtIpc()
    view = client.call("audio.workspace_get")
    assets = []
    try:
        catalogue = client.call("audio.design_catalogue")
        assert catalogue["max_duration_seconds"] == 600
        assert catalogue["working_pcm_budget_bytes"] == 512 * 1024 * 1024
        assert catalogue["processing_mode"] == "bounded_memory"
        assert {item["type"] for item in catalogue["layers"]} == {
            "oscillator", "noise", "fm", "resonator", "asset"}
        assert {item["type"] for item in catalogue["effects"]} == {
            "eq", "filter", "compressor", "delay", "reverb", "chorus", "phaser", "saturation", "limiter",
            "spectral_region"}
        with tempfile.TemporaryDirectory(prefix="rt_design_tools_", dir=args.output_dir) as text:
            folder = Path(text).resolve()
            source = folder / "source.wav"
            with wave.open(str(source), "wb") as wav:
                wav.setparams((2, 2, 48000, 0, "NONE", "not compressed"))
                wav.writeframes(b"".join(struct.pack("<hh",
                    int(10000 * math.sin(2 * math.pi * 440 * frame / 48000)),
                    int(7000 * math.sin(2 * math.pi * 880 * frame / 48000))) for frame in range(24000)))
            imported = client.call("audio.asset_import", path=str(source), license="test provenance")
            assets.append(imported["id"])
            # Cross both former limits (10 s synth / 30 s processed output).
            long_recipe = dict(mode="synth", preset="tone", duration_seconds=31,
                               frequency_hz=220, gain=0.2)
            long_render = client.call("audio.design_render", recipe=long_recipe)
            assert long_render["processed"]["frames"] == 31 * 48000
            assert long_render["processed"]["rms"] > 0
            for patch in ({"duration_seconds": 601}, {"duration_seconds": 400},
                          {"duration_seconds": 301, "stretch_ratio": 2}):
                rejected(client, "audio.design_render", recipe={**long_recipe, **patch})
                assert signature(client.call("audio.design_status")) == signature(long_render)
            long_saved = client.call("audio.design_save", folder=str(folder), name="Long source")
            assets.append(long_saved["asset"]["id"])
            long_path = Path(long_saved["design"]["path"])
            original_bytes = long_path.read_bytes()
            processed = client.call("audio.design_render", recipe=dict(
                mode="asset", asset_id=long_saved["asset"]["id"], source_out_seconds=31,
                effects=[{"type": "eq", "output_db": -3, "bands": []}]))
            assert processed["processed"]["frames"] == 31 * 48000
            edited = client.call("audio.design_save", folder=str(folder), name="Long edited")
            assets.append(edited["asset"]["id"])
            assert long_path.read_bytes() == original_bytes
            assert edited["asset"]["id"] != long_saved["asset"]["id"]
            assert signature(client.call("audio.design_reuse", id=edited["asset"]["id"])) == (
                processed["recipe"], processed["processed"]["pcm_hash"])
            track_id = None
            clip_id = None
            try:
                track_id = client.call("audio.track_create", name="Designer workflow test")["id"]
                before_clip = client.call("audio.list")
                rejected(client, "audio.clip_create", asset_id=edited["asset"]["id"],
                         track_id="missing_designer_test_track", start_seconds=0)
                assert client.call("audio.list") == before_clip
                # Retry only clip creation; no additional file or asset is needed.
                clip = client.call("audio.clip_create", asset_id=edited["asset"]["id"],
                                   track_id=track_id, start_seconds=0)
                clip_id = clip["id"]
                assert clip["track_id"] == track_id
                assert clip["asset_id"] == edited["asset"]["id"]
                assert client.call("audio.list")["assets"] == before_clip["assets"]
            finally:
                if clip_id is not None:
                    client.call("audio.clip_remove", id=clip_id)
                if track_id is not None:
                    client.call("audio.track_remove", id=track_id)
            scene = client.call("audio.list")
            base = dict(mode="synth", preset="tone", duration_seconds=0.5, seed=123)
            for kind in ("oscillator", "noise", "fm", "resonator", "asset"):
                layer = {"type": kind}
                if kind == "asset":
                    layer.update(asset_id=imported["id"], source_out_seconds=0.5)
                report = client.call("audio.design_render", recipe={**base, "layers": [layer]})
                assert report["processed"]["channels"] == 2
                assert report["processed"]["frames"] == 24000
                assert report["processed"]["rms"] > 0, f"Silent {kind}"
                again = client.call("audio.design_render", recipe=report["recipe"])
                assert signature(again) == signature(report), f"Non-deterministic {kind}"
                assert client.call("audio.list") == scene

            neutral = client.call("audio.design_render", recipe={**base, "layers": [{"type": "oscillator"}]})
            for effect in catalogue["effects"]:
                module = {"type": effect["type"]}
                if module["type"] == "eq":
                    module["bands"] = [{"frequency_hz": 220, "gain_db": 6}]
                if module["type"] == "limiter":
                    module["ceiling_db"] = -12
                recipe = {**neutral["recipe"], "effects": [module]}
                active = client.call("audio.design_render", recipe=recipe)
                assert math.isfinite(active["processed"]["peak"])
                assert active["processed"]["pcm_hash"] != neutral["processed"]["pcm_hash"], module
                bypass = client.call("audio.design_render", recipe={**recipe,
                    "effects": [{**module, "enabled": False}]})
                assert bypass["processed"]["pcm_hash"] == neutral["processed"]["pcm_hash"]
                zero_mix = client.call("audio.design_render", recipe={**recipe,
                    "effects": [{**module, "mix": 0}]})
                assert zero_mix["processed"]["pcm_hash"] == neutral["processed"]["pcm_hash"]
                visual = client.call("audio.design_effect_visual", effect=module)
                assert len(visual["points"]) >= 64 and all(
                    math.isfinite(value) for point in visual["points"] for value in point)

            # EQ curve and PCM must agree, including output trim and parallel mix.
            for mode in ("peak", "lowshelf", "highshelf", "highpass", "lowpass"):
                for hz in (20, 20000):
                    for q in (0.1, 10):
                        for gain in (-24, 24):
                            curve = client.call("audio.design_effect_visual", effect={"type": "eq",
                                "bands": [{"mode": mode, "frequency_hz": hz,
                                           "q": q, "gain_db": gain}]})
                            assert curve["sample_rate"] == 48000 and len(curve["band_curves"]) == 1
                            assert all(math.isfinite(value) for point in curve["points"] for value in point)
            quiet_path = folder / "eq_stereo_probe.wav"
            with wave.open(str(quiet_path), "wb") as wav:
                wav.setparams((2, 2, 48000, 0, "NONE", "not compressed"))
                wav.writeframes(b"".join(struct.pack("<hh",
                    int(800 * math.sin(2 * math.pi * 1000 * frame / 48000)), 0)
                    for frame in range(24000)))
            original_eq_bytes = quiet_path.read_bytes()
            quiet = client.call("audio.asset_import", path=str(quiet_path))
            assets.append(quiet["id"])
            eq_recipe = dict(mode="asset", asset_id=quiet["id"], source_out_seconds=0.5,
                             fade_in_seconds=0, fade_out_seconds=0)
            eq_neutral = client.call("audio.design_render", recipe=eq_recipe)
            probe = {"type": "eq", "bands": [{"frequency_hz": 1000, "gain_db": 6, "q": 0.707}]}
            boosted = client.call("audio.design_render", recipe={**eq_recipe, "effects": [probe]})
            ratio = boosted["processed"]["rms"] / eq_neutral["processed"]["rms"]
            assert abs(20 * math.log10(ratio) - 6) < 0.05
            for trim, mix in ((-6, 1), (6, 0.5)):
                effect = {"type": "eq", "bands": [], "output_db": trim, "mix": mix}
                expected = (1 - mix) + mix * 10 ** (trim / 20)
                rendered = client.call("audio.design_render", recipe={**eq_recipe, "effects": [effect]})
                ratio = rendered["processed"]["rms"] / eq_neutral["processed"]["rms"]
                assert abs(ratio - expected) < 1e-5
                curve = client.call("audio.design_effect_visual", effect=effect)
                assert all(abs(point[1] - 20 * math.log10(expected)) < 1e-8
                           for point in curve["points"])
            # A parallel high-pass mix has phase cancellation: dB/magnitude blending
            # would predict the wrong PCM level here.
            phase_probe = {"type": "eq", "mix": 0.5,
                           "bands": [{"mode": "highpass", "frequency_hz": 2000, "q": 0.707}]}
            phase_pcm = client.call("audio.design_render", recipe={**eq_recipe, "effects": [phase_probe]})
            points = client.call("audio.design_effect_visual", effect=phase_probe)["points"]
            upper = next(i for i, point in enumerate(points) if point[0] >= 1000)
            low, high = points[upper - 1], points[upper]
            fraction = math.log(1000 / low[0]) / math.log(high[0] / low[0])
            expected_db = low[1] + fraction * (high[1] - low[1])
            measured_db = 20 * math.log10(phase_pcm["processed"]["rms"] /
                                        eq_neutral["processed"]["rms"])
            assert abs(measured_db - expected_db) < 0.05, (measured_db, expected_db)
            client.call("audio.design_render", recipe={**eq_recipe, "effects": [probe]})
            exported = client.call("audio.design_save", path=str(folder / "eq_processed.wav"))
            assets.append(exported["asset"]["id"])
            assert exported["asset"]["id"] != quiet["id"]
            assert Path(exported["asset"]["path"]).resolve() != quiet_path.resolve()
            assert quiet_path.read_bytes() == original_eq_bytes
            recalled_eq = client.call("audio.design_reuse", id=exported["asset"]["id"])
            assert recalled_eq["recipe"]["asset_id"] == quiet["id"]
            assert recalled_eq["recipe"]["effects"][0]["type"] == "eq"
            assert recalled_eq["processed"]["pcm_hash"] == exported["design"]["pcm_hash"]
            data = Path(exported["asset"]["path"]).read_bytes()
            offset = 12
            while data[offset:offset + 4] != b"data":
                size = struct.unpack_from("<I", data, offset + 4)[0]
                offset += 8 + size + (size & 1)
            size = struct.unpack_from("<I", data, offset + 4)[0]
            samples = list(struct.iter_unpack("<ff", data[offset + 8:offset + 8 + size]))
            assert len(samples) == 24000 and all(right == 0 for left, right in samples)
            assert max(abs(left) for left, right in samples) > 0
            clipping = client.call("audio.design_render", recipe={**eq_recipe, "effects": [
                {"type": "eq", "bands": [{"frequency_hz": 1000, "gain_db": 24}] * 8}]})
            assert clipping["processing"]["clamped_samples_before_loop_join"] > 0
            assert clipping["processing"]["peak_before_output_clip"] > 1
            first = {"type": "eq", "bands": [{"mode": "lowpass", "frequency_hz": 300}]}
            second = {"type": "saturation", "drive": 0.8}
            ordered = client.call("audio.design_render", recipe={**neutral["recipe"], "effects": [first, second]})
            reversed_order = client.call("audio.design_render", recipe={**neutral["recipe"], "effects": [second, first]})
            assert ordered["processed"]["pcm_hash"] != reversed_order["processed"]["pcm_hash"]
            assert signature(client.call("audio.design_undo")) == signature(ordered)
            assert signature(client.call("audio.design_redo")) == signature(reversed_order)
            stable = client.call("audio.design_status")
            for patch in ({"layers": [{"type": "missing"}]}, {"effects": [{"type": "eq", "oops": 1}]},
                          {"effects": [{"type": "delay", "feedback": 1}]},
                          {"effects": [{"type": "eq", "bands": [{"q": 0}]}]},
                          {"effects": [{"type": "eq", "output_db": 25}]},
                          {"effects": [{"type": "eq", "output_db": "loud"}]},
                          {"effects": [{"type": "eq", "bands": [{"enabled": 1}]}]},
                          {"layers": [{"type": "noise", "enabled": 1}]},
                          {"effects": [{"type": "limiter"}] * 9}, {"gain_db": True},
                          {"tail_seconds": 6}, {"formant_compensate": 1},
                          {"gain": 0.5, "gain_db": 0}):
                rejected(client, "audio.design_render", recipe={**base, **patch})
                assert signature(client.call("audio.design_status")) == signature(stable)
            rejected(client, "audio.design_status", details=1)
            thin = client.call("audio.design_status", details=False)
            assert "processed" not in thin and thin["revision"] == stable["revision"]

            report = client.call("audio.design_analyze", fft_size=2048, points=256, time_bins=32, channel="left")
            assert report["continuous"] and report["hop_frames"] == 512
            assert len(report["spectrogram"]) == 32 and len(report["spectrogram"][0]) == 256
            assert report["pcm_hash"] == stable["processed"]["pcm_hash"]
            assert all(math.isfinite(value) for column in report["spectrogram"] for value in column)
            rejected(client, "audio.design_analyze", fft_size=1000)
            rejected(client, "audio.design_analyze", continuous=1)
            rejected(client, "audio.design_analyze", time_bins=129)
            waveform = client.call("audio.design_waveform", id=imported["id"],
                start_seconds=0.1, end_seconds=0.3, points=64)
            assert waveform["first_frame"] == 4800 and waveform["end_frame_exclusive"] == 14400
            assert waveform["points"][0]["first_frame"] == 4800
            assert waveform["points"][-1]["end_frame_exclusive"] == 14400
            assert waveform["channels"] == 2
            rejected(client, "audio.design_waveform", id=imported["id"], start_seconds=0.4, end_seconds=0.2)

            client.call("audio.design_render", recipe=dict(mode="asset", asset_id=imported["id"],
                source_out_seconds=0.5, fade_in_seconds=0, fade_out_seconds=0))
            for channel, expected in (("left", 440), ("right", 880)):
                measured = client.call("audio.design_analyze", fft_size=4096, channel=channel)
                dominant = max(measured["spectrum"], key=lambda point: point[1])[0]
                assert abs(dominant - expected) < expected * 0.05, (channel, dominant)
            # A short burst lies between sparse analysis windows. Full STFT must find it.
            burst_path = folder / "burst.wav"
            with wave.open(str(burst_path), "wb") as wav:
                wav.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
                wav.writeframes(b"".join(struct.pack("<h", int(10000 * math.sin(
                    2 * math.pi * 1000 * frame / 48000)) if 4560 <= frame < 4704 else 0)
                    for frame in range(24000)))
            burst = client.call("audio.asset_import", path=str(burst_path))
            assets.append(burst["id"])
            client.call("audio.design_render", recipe=dict(mode="asset", asset_id=burst["id"],
                source_out_seconds=0.5, fade_in_seconds=0, fade_out_seconds=0))
            sampled = client.call("audio.design_analyze", fft_size=1024, time_bins=4, continuous=False)
            full = client.call("audio.design_analyze", fft_size=1024, time_bins=4, continuous=True)
            assert max(point[1] for point in sampled["spectrum"]) < -100
            assert max(point[1] for point in full["spectrum"]) > -60

            # New recipe fields share the canonical renderer and survive normalization.
            level = client.call("audio.design_render", recipe={**base, "gain_db": -12})
            assert "gain_db" not in level["recipe"]
            assert abs(level["recipe"]["gain"] - 10 ** (-12 / 20)) < 1e-9
            assert signature(client.call("audio.design_render", recipe=level["recipe"])) == signature(level)
            asset_recipe = dict(mode="asset", asset_id=imported["id"], source_out_seconds=0.5,
                                fade_in_seconds=0, fade_out_seconds=0)
            forward = client.call("audio.design_render", recipe=asset_recipe)
            backward = client.call("audio.design_render", recipe={**asset_recipe, "reverse": True})
            assert backward["processed"]["frames"] == forward["processed"]["frames"]
            assert backward["processed"]["pcm_hash"] != forward["processed"]["pcm_hash"]
            formant = client.call("audio.design_render", recipe={**asset_recipe,
                "formant_semitones": 4, "formant_compensate": True, "formant_base_hz": 200})
            assert formant["processed"]["frames"] == forward["processed"]["frames"]
            assert math.isfinite(formant["processed"]["peak"]) and formant["processed"]["rms"] > 0

            tail = client.call("audio.design_render", recipe={**base, "tail_seconds": 0.25,
                "effects": [{"type": "delay", "time_seconds": 0.05, "feedback": 0.6}]})
            assert tail["processed"]["frames"] == 36000
            for preset in catalogue["presets"]:
                created = client.call("audio.design_render", recipe={**preset["recipe"], "duration_seconds": 0.5})
                assert created["processed"]["rms"] > 0
            # Save/reuse rebuilds the canonical recipe and the exact native preview PCM.
            saved = client.call("audio.design_save", folder=str(folder), name="Reusable sound")
            assets.append(saved["asset"]["id"])
            client.call("audio.design_render", recipe=base)
            recalled = client.call("audio.design_reuse", id=saved["asset"]["id"])
            assert recalled["processed"]["pcm_hash"] == saved["design"]["pcm_hash"]
            rejected(client, "audio.design_reuse", id=imported["id"])
            for patch in ({"designer_page": "bad"}, {"designer_zoom": 0},
                          {"designer_pan": 2}, {"designer_effect": 8},
                          {"designer_edit_view": "bad"}, {"designer_edit_view": 1},
                          {"designer_canvas_zoom": 0}, {"designer_canvas_zoom": 9},
                          {"designer_canvas_pan_x": -1}, {"designer_canvas_pan_y": 2},
                          {"designer_canvas_zoom": True}, {"designer_layer": 8}, {"designer_layer": -1}, {"designer_layer": True},
                          {"designer_tools_open": 1}, {"designer_properties_open": "yes"},
                          {"designer_eq_band": -1}, {"designer_eq_band": 8},
                          {"designer_eq_band": True}, {"designer_eq_range_db": 5},
                          {"designer_eq_range_db": 289}, {"designer_eq_range_db": True},
                          {"designer_eq_range_db": "24"}, {"designer_eq_auto_range": 1},
                          {"designer_panel_zoom": 0.4}, {"designer_panel_zoom": 1.6},
                          {"designer_panel_zoom": True}, {"designer_eq_height": 179},
                          {"designer_eq_height": 601}, {"designer_eq_height": "280"},
                          {"designer_panel_pan_x": -1}, {"designer_panel_pan_y": 2}):
                rejected(client, "audio.workspace_update", **patch)
                assert client.call("audio.workspace_get") == view
            client.call("audio.workspace_update", designer_page="effects", designer_zoom=4,
                        designer_pan=0.25, designer_effect=0, designer_layer=3, designer_tools_open=False,
                        designer_properties_open=True, designer_edit_view="arrange", designer_canvas_zoom=2,
                        designer_canvas_pan_x=0.25, designer_canvas_pan_y=0.75,
                        designer_eq_band=2, designer_eq_range_db=96, designer_eq_auto_range=False,
                        designer_panel_zoom=0.75, designer_eq_height=220,
                        designer_panel_pan_x=0.4, designer_panel_pan_y=0.6)
            updated_view = client.call("audio.workspace_get")
            assert updated_view["designer_page"] == "effects" and updated_view["designer_layer"] == 3
            assert not updated_view["designer_tools_open"] and updated_view["designer_properties_open"]
            assert updated_view["designer_edit_view"] == "arrange"
            assert updated_view["designer_canvas_zoom"] == 2
            assert updated_view["designer_canvas_pan_x"] == 0.25
            assert updated_view["designer_canvas_pan_y"] == 0.75
            assert updated_view["designer_eq_band"] == 2
            assert updated_view["designer_eq_range_db"] == 96
            assert not updated_view["designer_eq_auto_range"]
            assert updated_view["designer_panel_zoom"] == 0.75
            assert updated_view["designer_eq_height"] == 220
            assert updated_view["designer_panel_pan_x"] == 0.4
            assert updated_view["designer_panel_pan_y"] == 0.6
            for band_index, extent in ((0, 6), (7, 288)):
                bounded = client.call("audio.workspace_update", designer_eq_band=band_index,
                                      designer_eq_range_db=extent, designer_eq_auto_range=True)
                assert bounded["designer_eq_band"] == band_index
                assert bounded["designer_eq_range_db"] == extent
                assert bounded["designer_eq_auto_range"]
            for asset in reversed(assets):
                client.call("audio.asset_remove", id=asset)
            assets.clear()
        print("PASS: layers, rack order/bypass, visual curves, full FFT, source range, history and recipe reuse")
    finally:
        client.call("audio.design_stop")
        for asset in reversed(assets):
            client.call("audio.asset_remove", id=asset)
        client.call("audio.workspace_update", **view)
        client.close()


if __name__ == "__main__":
    main()
