# Post-processing and histogram exposure

Implemented 2026-09-05. The user reported that the application compiled and the
new controls worked on 2026-09-05. Codex did not build shaders/C++ or launch the
application. This is a general user confirmation, not a recorded cross-backend
image comparison or GPU timing measurement.

## Color pipeline

`source/include/PostProcess/ColorMath.h` is compiled directly by C++, CUDA and
GLSL. It contains the view transforms and grading math, rather than maintaining
three independent implementations.

Order: scene-linear Rec.709 HDR -> exposure -> Bradford white balance -> view
transform -> saturation -> look gamma -> vignette -> sRGB output encoding.
Exposure and white balance are linear operations before the nonlinear view
transform. No second grade is applied to an already display-encoded GPU surface,
including the final-render export path.

View transforms:

| ID | API name | Meaning |
| --- | --- | --- |
| 0 | `agx` | AgX approximation, Rec.2020 inset/outset, log2 encoding and contrast fit |
| 1 | `aces_fitted` | Matrix-based ACES fitted approximation; not a complete ACES/OCIO configuration |
| 2 | `uncharted` | Uncharted filmic curve |
| 3 | `filmic` | Hejl/Burgess-Dawson curve, linearized before output encoding |
| 4 | `linear` | No tone compression; clip to display range (`none` also resolves here) |
| 5 | `reinhard` | Explicit Reinhard compression |

As requested by the user, old AGX/ACES appearances are not preserved and None no
longer hides Reinhard. The default view is AgX. Old saved IDs use the new math.
`aces` remains an accepted spelling for `aces_fitted`, not a legacy operator.

White balance uses a daylight-locus source white and Bradford adaptation to D65.
6500 K is exactly neutral; higher Kelvin warms. Supported range is 4000–25000 K.
This is a deliberately bounded daylight model, not a full spectral illuminant
model. NaN, infinity and out-of-range public inputs are rejected; loaded scalar
settings are normalized to the supported ranges.

## Exposure

The main post panel owns the exposure mode:

- **Manual EV:** the post EV dial and optional linear Gain.
- **Physical Camera:** the same EV/gain plus the camera's calibrated ISO,
  shutter and f-stop multiplier. The camera's old auto-exposure flag does not
  override this mode.
- **Auto Histogram:** GPU HDR metering, bounded target EV and temporal adaptation,
  followed by the user's EV compensation and Gain.

The engine's light units are relative. Meter values are log2 scene luminance and
relative exposure stops, **not photometric EV100 or cd/m²**. +1 compensation stop
multiplies light by two.

Metering uses a deterministic grid of at most 128×128 HDR samples, 256 bins over
log2 luminance [-16,+16], and integer center weights. This bounds the GPU work
independently of render resolution. Black samples and non-finite radiance are
excluded. An empty histogram leaves the existing exposure unchanged.

Low/high percentiles trim the distribution with fractional boundary-bin weights.
Target exposure is `log2(key / metered_luminance)`, bounded by min/max EV.
Bright/dark adaptation has separate exponential rates in inverse seconds, not
frames. It operates in EV space and does not modify the user's Gain.

Vulkan records its histogram pass after HDR writes in the render command buffer.
Each existing frame-fence slot owns its descriptor and coherent 1 KB result buffer.
Only a completed slot is consumed. CUDA uses one pending 1 KB pinned-memory
transfer/event per render thread and skips rescheduling until it completes.
Neither meter downloads the full HDR image. Results arrive after GPU completion;
adaptation is intentionally delayed rather than synchronously stalling for a
current-frame histogram. CPU display uses the same histogram grid and reduction.

Changing post settings does not clear accumulation. Vulkan's trace push constant
exposure is fixed to one so camera exposure is no longer baked into the HDR
accumulation and then applied again at display time. OptiX's normal progressive
resolve and the OIDN CUDA resolve share the same display math.

**Raster/material preview shares the resolved exposure and color transform, but
does not produce its own HDR histogram.** Use Rendered mode for automatic
metering. `meter_supported`, `meter_valid` and `meter_source` expose this boundary;
the raster display-encoded image is never used as a fake HDR meter.

Lock Exposure captures the applied meter EV; `locked_ev` can also be set
explicitly. Final render freezes adaptation at the selected exposure, so a frame
or sequence does not vary with render duration. For repeatable exports, settle
the Rendered preview, lock exposure, and save the project before rendering.
Local exposure and full OCIO/display-device management are not implemented here.

## API / IPC

All new exposure operations go through `PostProcess/PostService.cpp`. UI,
scripting and IPC use the same service and validation. Existing post scalar
setters now live in the focused `Api/RtApiPost.cpp` module and are also reused by
the panel.

Python:

```python
rt.post.set(tone_mapping="agx", color_temperature=6500, gamma=1, saturation=1)
rt.post.configure_exposure(
    mode="auto_histogram", ev=0,
    min_ev=-12, max_ev=12, low_percent=2, high_percent=98,
    key=0.18, speed_up=3, speed_down=1, center_weight=0.5)
state = rt.post.get_exposure()
rt.post.configure_exposure(locked=True)
rt.post.reset_exposure()
```

IPC method/parameter examples (place inside the client's normal request envelope):

```json
{"method":"post.configure_exposure","params":{"settings":{"mode":"auto_histogram","ev":0,"center_weight":0.5}}}
{"method":"post.get_exposure","params":{}}
{"method":"post.configure_exposure","params":{"settings":{"locked":true}}}
{"method":"post.reset_exposure","params":{}}
```

Configuration patches are atomic: unknown names, invalid types, non-finite
numbers, invalid percentile ordering or unsupported bounds reject the whole
patch. Final renders reject mutations. Python raises an exception; IPC returns
the existing `__error`/`invalid_parameter` result. Reset discards stale in-flight
measurements using a generation ID and preserves the explicit lock setting.

`post.get` reports the user Gain. `post.get_exposure` reports configuration,
target/applied meter EV, measured luminance, histogram and resolved display
transport separately. New settings persist under `postfx.exposure_v2`.

## Verification

Completed without building:

- Shared color functions interpreted numerically in the tool's JavaScript runtime:
  gray ramps over ±20 stops, all six transforms, saturated primaries, invalid/
  extreme values, neutral white balance and warm-white direction passed.
- AgX gray input 0.18 produced approximately (0.21455, 0.21450, 0.21450).
  Inputs 2 and 10 remained distinct (green outputs approximately 0.74329 and
  0.95526); the old implementation clipped both to one.
- The actual C++ adaptation expression, interpreted numerically, gave the same
  result for one second and sixty 1/60-second steps within floating-point tolerance.
- `python scripts/audit_ipc_capabilities.py` passed with current generated
  descriptors, including all three new exposure operations.

`scripts/check_post_v2.mjs` contains reproducible non-build color/temporal/source
checks for a machine with Node installed. Node was unavailable on this agent's
PATH; equivalent shared-math checks were run using the tool runtime instead.

`RayTrophiStudio/tests/post_exposure_tests.cpp` is a standalone C++ contract test,
not an application source and not included in the Studio target. It checks the
actual exposure core: validation, empty histograms, firefly rejection, clamps,
time-step invariance, stale-generation rejection, freeze/lock and transform
semantics. Keep it under `tests`; deleting it does not change application behavior.
It was not compiled or executed by Codex.

User-run checks after building:

1. Rebuild shaders with the project's normal shader workflow (including
   `exposure_histogram.comp`, `tonemap.comp`, and raster shaders including
   `post_chain.glsl`), then build the Studio target. VS and CMake registrations
   include the new PostProcess modules. The user has reported a successful build.
2. In Rendered mode, select AgX and Auto Histogram. Move between dark and bright
   views. Confirm target/applied EV, histogram and source update; no abrupt
   per-frame brightness oscillation. Test Vulkan and OptiX separately.
3. After max samples, change EV, view transform and white balance. The picture
   must refresh without adding samples or restarting accumulation.
4. Lock exposure, change lighting, and confirm applied meter EV remains fixed.
   Save/reopen and check lock, bounds, weighting and mode round-trip.
5. With fixed Gain/EV/camera and denoising disabled, compare CPU/Vulkan/OptiX
   neutral ramps and saturated emissive colors. Allow small rasterization,
   sampling and output-quantization differences; investigate systematic exposure
   or gamma differences. Toggle OIDN and check for double grading.
6. Export the same locked final frame twice and a locked animation sequence.
   Confirm display appearance matches the viewport and does not depend on render
   duration. Confirm transparent-background alpha is retained.
7. Submit invalid settings through script and IPC; compare state before/after to
   confirm atomic rejection. Inspect `agent.describe` for the new methods.

To run the optional standalone C++ test from a Visual Studio developer terminal,
compile `tests/post_exposure_tests.cpp` together with
`source/src/PostProcess/Exposure.cpp`, using C++17 or later and
`source/include` as an include directory, then run the resulting executable.
This is separate from the normal Studio build.

## References

- [three.js shared tone-mapping implementation](https://github.com/mrdoob/three.js/blob/dev/src/renderers/shaders/ShaderChunk/tonemapping_pars_fragment.glsl.js):
  AgX Rec.2020 matrices/log range and polynomial, and matrix-based ACES fit.
  Original three.js code is MIT licensed; numerical formula adaptation is
  attributed here and in `ColorMath.h`. ACES fit originates with Stephen Hill;
  AgX implementation traces through Filament to Blender.
- [Epic auto exposure documentation](https://dev.epicgames.com/documentation/en-us/unreal-engine/auto-exposure-in-unreal-engine):
  histogram metering, center weighting and temporal exposure concepts.
- [OpenColorIO](https://github.com/AcademySoftwareFoundation/OpenColorIO): future
  full color-management integration; not a dependency of this implementation.
