# Gas/VDB material detail audit and ramp packing repair

Status: source inspection and first repair complete; user build/numerical/live
acceptance pending. No application, C++ build or shader compiler was launched.

## What happens to small differences?

Small differences are not categorically rounded away. The inspected live gas
upload uses `cells * sizeof(float)` in `VulkanBackend_Volumes.cpp`; its CUDA
texture path also uses float channels. Vulkan dense temperature sampling uses
trilinear interpolation and the historical `solver heat * 3000` conversion.
That conversion is not an explicit physical-unit contract, and it must not be
silently replaced by an ambient offset without auditing every producer.
This inspection does not certify the precision of every imported VDB format.

The transfer and transport contain the following detail-loss mechanisms:

| Stage | Current behaviour | Implication |
| --- | --- | --- |
| Temperature interval | Clamp to authored min/max before colour evaluation | Distinct out-of-range values become identical |
| Ramp coordinate | Clamp normalized temperature times temperature scale to 0..1 | Scale above one can flatten an extended hot region to the last stop |
| Blackbody colour | Continuous log/power RGB fit, input clamped to 1000..40000 | Approximation, not spectral integration; colour changes can be subtle |
| Blackbody strength | `(clamped_temperature / authored_max)^4` | Strongly suppresses cooler emission; normalized artistic law, not calibrated visible-band radiometry |
| Ramp strength | No corresponding temperature power; RGB times density/intensity | Similar ramp RGB values produce similar source strength despite temperature differences |
| Ramp alpha | Stored on CPU; not present in the GPU RGB ramp payload | Alpha is not an independent GPU emission intensity curve |
| Density remap | max((d-low)/(high-low), 0), followed by cutoff | Values at/below cutoff vanish; above cutoff a feather fades in over cutoff..2*cutoff |
| Radiance limiter | Luminance capped at 64 by scaling RGB | Once capped, hotter/brighter samples can lose amplitude contrast while hue survives |
| Integration | Finite march steps, trilinear reconstruction, accumulated extinction | Fine features can be blurred or missed; optically thick foreground naturally hides interior detail |

Vulkan RT and RayFusion's actual volume fragment shader both contain the T^4
blackbody branch and the flat-strength ramp branch. A previous conversation
compared `ray_color.cuh` with Vulkan and called that RayFusion; that CUDA file is
the OptiX path, not RayFusion. Correct parity inspection uses
`material_preview_volume.frag` plus `material_preview_volume_fields.glsl`.
The presence of the same equations does not establish visual or sampling parity.

The RT shader also retains a density fallback for nonpositive temperature,
whereas the RayFusion volume fragment rejects an explicitly cold temperature
field when no flame is present. VDB unit detection still uses a per-sample >20
heuristic outside the flagged live dense-gas path. Both need explicit source-unit
and missing-channel semantics in a later increment.

For example, with an authored interval 1000..5000, 6000 and 9000 both clamp to
5000. With a ramp coordinate of 0.7 and temperature scale 2, both 0.7 and 0.9
end up at coordinate 1.0. No extra grid precision can recover those differences
after this mapping. Conversely, two in-range temperatures do not necessarily
need visibly different hues: brightness, exposure and optical depth matter too.

## Repair delivered in this increment

The UI permits more than eight emission ramp stops. `VolumeShader::toGPU` and
two gas-specific packing sites previously took the first eight and discarded
all remaining stops. This could remove the entire hot end of the authored ramp.

`VolumeRampPacking.h` now provides the canonical packet builder. Ramps with
0..8 stops retain exact values and order. Larger finite ramps are sorted in a
local copy, retain both endpoints, and greedily retain the remaining knots with
the greatest maximum RGB interpolation error. The existing eight-stop ABI stays
unchanged. Ordinary <=8-stop ramps, including the five-stop default, are unchanged.
All VDB and gas packet construction now uses `VolumeShader::toGPU`; the gas paths
copy that canonical result instead of rebuilding a truncated ramp.

This is still an approximation for complex ramps above eight stops. It does not
promise exact reproduction, optimal error, alpha evaluation or recovery of every
narrow peak. It fixes silent tail truncation without adding a GPU buffer or
changing existing scene data. CPU source stops and scene serialization remain
intact. Existing UI/script/IPC mutation operations and validation are unchanged;
this is a derived-render-data repair, not a new authoring operation.

## Next implementation order

1. Independent temperature-to-intensity response, defaulting to legacy behaviour
   for existing files. Do not repurpose previously ignored ramp alpha and
   silently change old fire scenes. Deliver UI, Python, IPC and validation together.
2. Explicit source temperature units and fallback policy, shared by RT, RayFusion,
   emitter sampling and CPU evaluation. Audit normalized heat versus actual Kelvin.
3. Density curve/gamma and controlled ignition/soft threshold, sharing evaluation
   with shadow queries and conservative majorants.
4. Optional full ramp LUT/storage if more than eight stops must be reproduced
   faithfully; inspect performance and resource lifetime before changing the ABI.
5. Audit radiance limiting and tone mapping with measured sample histograms;
   do not simply remove the firefly limiter to chase contrast.

## User verification checklist

- Rebuild C++; this increment changes no shader source or GPU ABI.
- Compile/run `scripts/test/volume_ramp_packing_test.cpp` as C++17 with
  `RayTrophiStudio/source/include` on the include path and assertions enabled.
  It is standalone and does not need renderer or Vec3 linkage.
- In both Vulkan RT and RayFusion, compare the unchanged default five-stop ramp.
- Test a >8-stop ramp with a distinct colour/peak at stop 9 and a distinct last
  endpoint: the hot end must no longer freeze at the eighth authored stop.
- Save/reload: all original CPU stops must remain present. Compare an imported
  VDB and a live gas domain; both must use the same selected packet knots.
- For detail diagnosis keep camera/lighting/exposure fixed, choose a temperature
  interval spanning the field, set temperature scale to 1, and compare low
  intensity against the current setting. This isolates clamping and tone mapping
  from a change in simulation data. No live scene settings were changed here.

Non-build checks: targeted whitespace check and packing/wiring/ABI source audit.
The new C++ numerical test is supplied but has not been compiled or executed.
