# SatMap preset recipes

Each JSON file describes one data-driven SatMap graph recipe. Applying a recipe
creates ColorRamp, condition-remap, paint-resolution mask-combine and SatMap
Blend nodes. The created nodes store full ramp/property snapshots in the terrain
graph, so saved projects do not depend on the source preset file remaining
installed.

Supported fields:

- `height`, `slope`, `concavity`, `convexity`, `valley`, `wetness`, `exposure`
- `flow`, `channel_width`, `erosion`, `deposition`
- `soil`, `grass`, `cavity`, `mud`, `moss`

`channel_width`, `erosion` and `deposition` require an authored Hydraulic
Erosion node. A layer that requests an unavailable field is skipped and reported
as a warning; the system never substitutes a semantically different field.

Every layer has one color-driving `primary` field and zero or more coverage
`conditions`. Conditions are multiplied at paint resolution, even when their
source fields have different native resolutions. `invert: true` selects the
lower side of the condition range.

Ramp stops are `[position, red, green, blue]` or
`[position, red, green, blue, alpha]`, with values in `[0, 1]`. Between 2 and 32
stops are accepted.
