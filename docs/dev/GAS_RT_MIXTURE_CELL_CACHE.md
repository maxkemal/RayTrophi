# Gas RT: quality-preserving cell-bound cache

2026-10-29. Source change; GPU acceptance awaits the user's shader build.

## Observed cost

At frame 89, paused, 1680 x 945, the existing scene measured approximately
5.3 ms with both optical fields hidden, 23.9 / 20.6 ms with either field alone,
and 189.3 ms with both visible. These are short progressive GPU captures,
not a controlled final benchmark. Disabling adaptive stepping reduced the
two-field measurement to 53.8 ms; disabling shadows alone left 145.9 ms.
The overlapping adaptive mixture path is the primary suspect. All material
settings, visibility and physical field hashes were restored after probing.

## Implementation

`volume_mixture_cell_cache.glsl` retains the directional raw-density slope
bound and largest corner value for the last voxel of each of the first two
prepared NanoVDB contributors. The original eight corner reads and bound
calculation run on a cache miss. Cache hits reuse only these scalar bounds.
The camera's fixed affine ray, box interval and material curve slope are also
prepared once. Additional contributors retain the ordinary calculation.

The cache resets for every mixture invocation. Random-walk segments bypass
it because their direction changes. Invalid corner values are never cached.
Only two scalars per contributor are retained instead of eight corner values
to limit additional shader register pressure.

Sample positions, adaptive error target, cell exits, density interpolation,
emission limits, authored step budget, random numbers and shadow calculation
are unchanged by the cell cache. No sampled density, lighting, radiance or
selected step length is cached. There is no new material option or ABI field.

## Verification and remaining acceptance

- `python scripts/test/check_volume_mixture_adaptive_contracts.py`: 12 tests pass.
  Includes exact cached/uncached analytic march-sequence and transmission
  equality, reduced corner queries, cache lifetime and walking fallback.
- `python scripts/test/check_volume_mixture_step_budget.py`: 4 tests pass.
- These are non-build checks; they do not establish compiled GPU image parity
  or a measured speedup. Shader compilation is performed by the user.

Rebuild the RT shaders through the existing shader workflow, including
`volume_closesthit.rchit`, then keep the current scene at frame 89 / Pause.
Repeat hidden / single / both-visible GPU captures at identical resolution,
camera, accumulation range and material settings. Compare converged images
with the previous shader at the same seed and sample count. Check transformed
volumes, three overlapping fields and random walk as regression cases.
Confirm speedup and image parity before declaring this performance issue closed.
