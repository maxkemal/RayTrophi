# Terrain Node Contract Redesign

> **Durum: AKTIF** — Phase 1 infrastructure implemented and VERIFIED live over
> IPC (checks 1-7 and 9 in `NEXT_BUILD_CHECKS.md`). Hydraulic Erosion is
> physically pruned to 4 outputs and its consumers moved to stable-key lookup.
> Driving the checks found two more consumer defects in
> `createSnowyMountainValleyGraph`; both are fixed and awaiting a rebuild.

## Product rule

A terrain port is public only when it carries an authoritative result that materially changes a downstream decision. Solver bookkeeping, weak classifications, normalized previews and data that belongs to another domain are not equal authoring choices.

Every port is classified as:

- **Primary**: the node's essential authoring contract; always visible.
- **Optional**: a meaningful expert override or reusable measured field; exposed from Properties and always visible while connected.
- **Diagnostic**: inspection data, not a normal graph dependency. Diagnostic ports are transitional and should become property previews or be removed.

Default target: no more than three visible inputs and four visible outputs. Expert river/output sinks may exceed this only through connected optional ports.

## Implemented foundation

- Pins have stable keys, sections and exposure tiers.
- Terrain nodes receive their presentation profile after graph registration, covering UI and registry/script-created nodes through the same path.
- Properties exposes Optional and Diagnostic ports. Connected ports cannot be hidden.
- Port visibility and stable keys serialize with the graph. Loading resolves stable keys before falling back to legacy array order.
- Script and IPC expose `nodes.list_ports`, `nodes.set_port_visible`, and `nodes.link_by_key`.
- `GraphBase::addLink` makes both endpoints visible, preventing cables without sockets.

## Hydraulic Erosion target contract

### Primary

Inputs:

- `height`
- `area`

Outputs:

- `height`
- `wear`
- `deposits`
- `flow`

`flow` is the artist-facing sediment transport path. It must not be confused with physical discharge in cubic metres per second.

### Move to authoritative owners

| Current Hydraulic product | Target owner | Decision |
|---|---|---|
| Discharge | Watershed/River Hydraulics | Remove from Hydraulic public contract after setup migration |
| Flow Direction | Watershed Analysis | Remove duplicate |
| Drainage Area | Watershed Analysis | Remove duplicate |
| Channel Width | River Hydraulics | Remove duplicate |
| Water Depth | River Hydraulics | Remove duplicate |
| Water Level | River Hydraulics | Remove duplicate |
| Lake Depth | Lake Basin | Remove duplicate |
| Grain Size | Sediment Select source | Keep only if material/biome workflows demonstrate a real consumer |
| Gravel/Sand/Silt | Sediment Select | Remove; these are thresholds of one grain-size field |

`HydraulicErosionFields` may retain physical arrays internally. Removing a graph pin does not mean making the solver less physical; it means publishing each measurement from one authoritative node.

## High-density node decisions

| Node | Primary contract | Optional / follow-up |
|---|---|---|
| Thermal Erosion | Height, Area -> Height, Wear, Deposits | Hardness optional; review Talus; derive/remove Rock Exposure |
| Noise Generator | Height | Macro/Ridge/Valley remain optional until consumer audit |
| Mountain Range | Base, Area -> Height, Ridge, Valley | Uplift/Hardness/Fracture need consumer audit |
| Plate Tectonics | Height, Boundary, Uplift | Crust ID diagnostic; Ridge/Valley duplicates reviewed |
| Watershed Analysis | Height, Rain -> Conditioned Height, Flow, Basins | Direction, catchment, runoff and breach remain expert physical data |
| Lake Basin | Height -> Lake Mask, Depth, Level | Shoreline/spill are expert; Lake IDs diagnostic |
| River Network | Flow -> Channels, Order, Sources | Lake constraints optional |
| River Hydraulics | Bed, Catchment, Channels -> Discharge, Speed, Level | Froude/Foam diagnostic unless water shading proves consumers |
| River Bed Carve | Height, Channels, Width, Depth -> Height, Bed | Reference/lake/glacial/breach constraints optional |
| River Spline Output | Height, Flow, Direction, Channels | Rendering fields visible only when connected |
| Surface Composer | Height, Soil, Flow -> Splat | Climate/geology influences optional |
| Biome Composer | Height -> Biome Splat | Individual biome masks optional |
| Terrain Fields Output | Connected fields only | Replace fixed 38-port face with Add Field, later consider one Publish Field node per field |
| Foliage Set | Connected layers plus next empty slot | Variadic UI instead of eight permanent sockets |

## Node removal/merge candidates

These are not removed until their factory, palette, setup, script/IPC and actual-consumer audit agree:

- `ErosionWizard`: already a legacy passthrough and registry exception. Remove the remaining enum/load compatibility branch when no current setup creates it.
- `Overlay` and `Screen`: fold into `Blend` modes if their mask/range behavior is identical.
- `AutoSplat` versus `SurfaceComposer`: keep an Easy/Expert distinction only if both own different workflows; otherwise retain Surface Composer and make presets provide the easy path.
- `Normalize`, `Clamp`, `Remap`, `MaskAdjust`: consolidate only after verifying morphology/range semantics. Similar names are not sufficient evidence.
- Standalone `HardnessInput/Output`: keep while hardness is an explicit persisted terrain field; remove only if the field publication contract replaces both operations.

## Missing focused nodes

Only add these when the corresponding output survives the consumer audit:

- `Sediment Select`: one grain-size/transport input, range controls, one mask output. Replaces Gravel/Sand/Silt pins.
- `Publish Field`: name + one typed field input. Candidate replacement for the 38-input Terrain Fields Output sink.
- `Field Select` only if a typed terrain-field bundle is later introduced. Do not add an opaque bundle merely to hide cables.

## Batch log

**2026-08-29 — Hydraulic batch closed for build.** Removing the nine-output
face left three consumers reading slots that no longer mean what they meant:

- `TerrainSatMapPresetLibrary.cpp` read slot 6 for `channel_width` behind a
  `size() > 6` guard. The guard can never be true again, so the field went
  silently to zero. Channel width is metres of wetted river and its
  authoritative owner is River Hydraulics: the recipe field now reads that
  node's `river_width` port, and is 0 only when no River Hydraulics node
  exists.
- `TerrainSatMapPresetLibrary.cpp` and `TerrainSatMapSetup.cpp` read slot 3 as
  `flowSource`. That slot was Discharge and is now Flow. Both are 0..1
  normalised fields with river-shaped support, so the swap costs nothing at
  compile time and everything downstream. Both sites now look the port up by
  the `flow` key.
- Graph loading fell back to the saved index whenever a key did not resolve,
  which is right while a contract grows and wrong once one is pruned. The
  fallback is now disabled for a saved list longer than the node's current one,
  a per-type legacy slot table maps the ports that survived (old Sediment Flux
  -> new Flow) and drops the ones that moved to other owners, and the load
  reports how many saved ports it could not restore instead of quietly opening
  with fewer links.

A fourth consumer was found in the measurement layer rather than the graph:
`terrain.flow_authority` classified River Network's source as `"hydraulic"`
when both its area and direction came from Hydraulic Erosion. That arm can
never be true again, so it was removed and the case now reports `"mixed"`,
which is the honest answer for a network fed from Hydraulic's remaining Flow
pin. `terrain_mask_pipeline_check.py` was updated in both script copies.

★ That removal exposed something the pruning has not decided yet: the
**2-channel vector flow direction only ever existed on Hydraulic Erosion**.
Watershed Analysis publishes a 1-channel direction code, so the pipeline
check's `river_direction_channels == 2` assertion is no longer satisfiable by
any wiring and was not promoted to the watershed path. Before the Watershed
batch, decide whether Watershed owes the graph a vector direction or whether
the D8 code is the contract and that assertion should be rewritten against it.
Until then the "one paired authority" check stands and the channel assertion
is simply not made.

**Two further consumers surfaced only by driving the app**, and neither would
have been found by reading:

- `createSnowyMountainValleyGraph` read `erosion->outputs[5]` -- past the end of
  a four-element vector. It did not crash: the out-of-bounds id failed pin
  lookup and only the setup fault reporter made it visible. Slot 5 was the
  2-channel vector flow direction, and measuring it settled the open question
  above from the other side: Watershed's 1-channel direction is REFUSED by
  Surface Relief's Flow Direction pin (tested live), so that input now has no
  producer at all and Surface Relief always falls back to the downhill
  gradient. The link was removed; the ownership decision stays open.
- The same setup fed `erosion->outputs[3]` into `FlowMask`'s **Discharge**
  input. Slot 3 was Discharge in m3/s and is now Flow, log-normalised sediment
  transport. FlowMask republishes that pin verbatim as the canonical flow
  magnitude and sets `lastDischargeMeasured` from the pin being CONNECTED, not
  from what it carries -- so `terrain.flow_authority` reported
  `source="measured"` over a field that is not discharge. Measured live on the
  snowy preset before the fix. An authority that grades a cable is not an
  authority. Removing the stale link also unblocks the intended wiring: the
  river setup's `wireFlowAuthority` only fills an EMPTY pin, so the leftover
  link had been keeping Watershed's field out of it.

**Measured after the rebuild.** Removing the stale wire let the river setup try
its intended link, and the attempt was REFUSED:
`Watershed Analysis.Accumulation -> Flow.Discharge: would create a cycle`.
Isolated by comparing two graphs -- `default + river_network` (no Surface
Relief) connects that pin cleanly, `snowy + river_network` cannot. The cause is
structural, not a wiring slip: **Flow occupies two positions in the DAG at
once.** Surface Relief consumes its channel mask as micro-relief input, so Flow
sits ABOVE the final height; hydrology wants to feed Flow from a Watershed
computed FROM that final height, which needs Flow BELOW it. The same preset was
already reporting two other cycle faults on the Surface Relief / Flow / Soil
Depth triangle, so this is one tangle, newly visible rather than newly created.
**Resolved in two rounds.** Round one: neither of the two obvious fixes was
needed --
the setup already peels Snow and River Bed Carve off the authored height to
find the hydrology ground, and Surface Relief simply was not on that list. It
is now. Hydrology reads the macro surface; the trunk is
`Erosion -> hydrology -> Flow -> Relief -> Snow -> Carve -> Height Output`.

This makes the code agree with a rule the landform setup already wrote down --
"Surface Relief reads it but does not feed back into the drainage solve; its
sub-cell-scale geometry cannot invalidate the macro river network" -- while
the hydrology branch was in fact reading Relief's output as its ground.

Only the hydrology tap moved. Terrain Analysis still classifies the detailed
surface, because that is the surface being shaded; moving it as well would
have restyled every splat and biome mask under cover of a cycle fix.

Round two, measured rather than guessed. Peeling Relief cleared two of the
three cycle faults; the third survived. Saving the graph and walking the link
table upstream from Watershed named the remaining path exactly:

```
Flow.Discharge        -> Surface Relief.Flow (optional)
Surface Relief.Height -> Snow.Base Height
Snow.Meltwater Depth  -> Watershed.Water Input Depth
```

The lower two edges are physically right -- snow falls on the detailed
surface, meltwater feeds the drainage. The top edge was wrong on its own
merits, cycle aside: `FlowMask.Discharge` publishes the physical magnitude,
drainage area in SQUARE METRES, 1e4..1e6, while Surface Relief spends it as
`clamp01(flowValue * flowInfluence * ...)`. The rill claim was therefore
saturated across the entire map. The river setup warns about precisely this
two nodes away -- "feeding the physical Discharge sibling here is both a unit
mismatch and an immediate saturation of almost every real river cell" -- and
Surface Relief was doing it.

Rills now read `Hydraulic.flow`, log-normalised to 0..1 by construction and
sitting above the whole hydrology branch, which fixes the units and the
ordering with one link. It is also the truer field: rills are erosion
features, so they belong to sediment transport rather than to the macro
drainage network the rivers are cut from.

★ Worth carrying forward: a unit mismatch and a cycle turned out to be the
same wire. The cycle was the loud symptom; the saturated rills had been silent
since the pin was added.

Round three, same pattern one node over: `Flow.Channel -> Soil Depth.Flow`
still cycled, because Soil Depth is upstream of hydrology too (Soil -> Surface
Relief -> Snow -> meltwater -> Watershed). The landform setup already feeds
that pin from `Hydraulic.flow`, which is 0..1 and acyclic, so the river setup
was trying to repoint a correctly wired input. Guarded with the authored-wins
rule `wireFlowAuthority` uses one screen up: fill the pin only when it is
empty.

**Measured after the fix**, at 1024 m / 512 (2 m cells), comparing
high-frequency roughness of the baked heightfield in channel cells against dry
cells: ratio **3.44**, detail covering **19.5%** of the map -- rills are
confined, not spread. Driving `rockAmplitudeMeters` from 0.85 to 25 reproduces
what a saturated claim looks like (ratio 1.31, coverage 43.5%) and restoring it
returns 3.44 / 19.5% exactly, so the metric distinguishes the two states.

Two measurement notes worth keeping, both of which cost a false result first:

- Resolution is not incidental. Surface Relief clamps its feature size to the
  representation cell, so at 512 m / 128 (4 m cells) the authored 13 m feature
  is pushed to 16 m with 4 samples across it. The effect cannot exist there;
  an eyeball check at that resolution reports nothing and means nothing.
- The standalone `_flow_*.png` export has a mean of 0.22 out of 255, so
  quantile thresholds over it select noise. The first run of the measurement
  reported a ratio of exactly 1.00 from it -- a broken instrument, not a
  result. Flow must come from channel R of the surface-semantic map.

A saturation control can no longer be built at all: `flowInfluence` clamps to
[0, 2] on write and the field it multiplies is now 0..1, with dry cells at ~0,
so no multiplier saturates the map. The old behaviour needed the INPUT to be
1e4..1e6. That the control is now unbuildable is the fix, not a gap.

Left deliberately unchanged: `wireFlowAuthority` feeds `FlowMask.discharge`
from Watershed **Accumulation**, which is Unitless, so `discharge_measured`
means "Flow reads a solver field" and not "Flow reads m3/s". That is the
pre-existing behaviour and `default + river_network` already reported it. The
true m3/s owner, `RiverHydraulics.discharge`, is acyclic from here and is the
better long-term source -- but FlowMask thresholds its input into a channel
mask and the two fields' ranges differ by orders of magnitude, so that swap
needs a threshold recalibration and its own batch.

## SatMap port face: measured, then addressed

Asked whether SatMap should simply derive everything from height. Measured on a
`snowy + river_network + river_network_detailed` graph first:

| Node type | nodes | input pins | connected | idle | pins carrying an exposure tier |
|---|---|---|---|---|---|
| `Terrain.SatMapColorRamp` | 5 | 45 | 12 | **33** | **0** |
| `TerrainV2.TerrainFieldsOutput` | 1 | 36 | 23 | 13 | 36 |
| `TerrainV2.Remap` | 12 | 24 | 12 | 12 | 0 |

Two findings, and neither argues for deriving from height:

1. **The whole SatMap family sits outside the port profile.** Its type ids live
   under `Terrain.`, and `configureTerrainNodePorts` only branches on
   `TerrainV2.`, so every SatMap pin reports `primary` and nothing can be
   tiered away. `TerrainFieldsOutput` shows the mechanism works where it is
   applied: 36 pins, all tiered, face collapses to the connected ones.
2. **One class serves two roles.** The base ramp uses 8 of its 9 inputs; the
   four per-layer ramps are built with `autoDeriveMasks=false` and every blend
   weight zeroed, so they consume exactly one field each and still draw nine
   sockets. That is where 33 idle pins come from -- not from too many sources.

Deriving the fields from height instead would be the wrong reduction twice
over. The fields that are height-derivable (slope, concavity, convexity,
valley, wetness) already have one authoritative producer in Terrain Analysis,
and re-deriving them inside SatMap would put a second, disagreeing producer in
the graph -- the exact shape this redesign exists to remove. The rest are not
in the height at all: soil depth is deposition history, hardness is lithology,
snow/ice is climate state, channel width is Manning over discharge. A field
derived from height and given one of those names is a guess wearing a
measurement's label.

The reduction that fits the product rule instead: give the SatMap family the
same exposure tiers. Done, and not with 62 hand-written opinions -- with a
default profile that reads what each node's author already declared:

- input 0 is the node's subject and stays Primary. Measured across all 65
  terrain node classes, the first input is always the thing the node acts on
  (Height, Base Height, Bed Height, Mask, In, A, Source, Splat, Accumulation).
  There is no counter-example, so this is a convention, not a guess.
- every other input constructed with `optional = true` becomes Optional:
  hidden until connected, always listed in Properties.
- a required input stays Primary. Hiding one would hide an error.
- outputs are untouched. An unread output is a question about consumers, and
  hiding it would answer that question by making it invisible.

Both failure modes are mild, which is what makes a mechanical rule safe here:
a pin wrongly tiered Optional still appears the moment it is connected and is
always reachable from Properties, and a pin wrongly left Primary just keeps
today's behaviour.

Letting the recipe drive the face as well -- it already declares every field it
reads through `recipeUses`, `layer.primary` and each condition's `field` -- is
still open, and is now a refinement rather than the fix.

## Dead pin removed: Surface Relief's flow direction

The open ownership question above resolved itself into a deletion once it was
measured properly. Enumerating every Direction pin in the terrain set:

- exactly one Direction OUTPUT exists, Watershed Analysis's, at **1 channel**
- Surface Relief's input was the only one demanding **2**

So no wiring could ever satisfy it, the node fell back to its downhill gradient
on every evaluation, and a live link attempt was refused. The pin is gone. The
gradient is computed from the height the node already reads, so nothing was
measured before that is unmeasured now -- only a socket that could never be
filled has stopped being offered as an authoring choice.

The audit that used to REQUIRE that pin has been replaced by the invariant that
makes the state unreachable: no input may demand a channel count that no output
of the same semantic can supply. All three of the audit's new rules were proven
to fail before being trusted.

## Measurement honesty: slope_area_fit

Not part of the port contract, but found by it and worth recording next to it.
`terrain.slope_area_fit` reads `terrain->flowMap`, and that buffer is filled by
`terrain.calculate_flow` **only** -- evaluating the node graph does not fill it.
Skip the call and the buffer is allocated, correctly sized and entirely zero,
so it passed the size guard and every cell fell below the channel threshold.
The answer was `too_few_channels`: a sentence that reads as a measured fact
about the landscape while meaning "the buffer I read was empty". Measured back
to back on one terrain: **0 -> 13024**. It cost a false regression report.

The descriptor overlay was telling agents the opposite -- "needs a computed
flow field (terrain.calculate_flow **or a graph evaluation**)". Corrected.

Now `flow_peak` reports what the fit actually read and `flow_field_empty` names
the empty-input case separately from a genuinely sparse network.

The general lesson for the remaining pruning batches: a removed pin does not
announce itself at the consumer. Every `outputs[n]` on a pruned node is either
a compile error (best case), an out-of-range guard that is now permanently
false (silent zero), or an in-range slot that now carries a different
measurement (silent wrong). Only the first one is loud, and it is the rarest.

## Setup migration

1. Replace numeric `inputs[n]`/`outputs[n]` authoring with stable port-key lookup.
2. Build the visible trunk as `Landform -> Erosion -> Carve -> Snow/Relief -> Height Output`.
3. Route hydrology only through Watershed, Lake, Network and River Hydraulics owners.
4. Stop publishing the same physical fact from Hydraulic and Watershed simultaneously.
5. Remove setup links to Diagnostic ports.
6. Re-layout groups after the actual link count falls; hiding cables is not a substitute for deleting false dependencies.

## Required verification per pruning batch

- Node reachability audit passes.
- IPC capability/descriptor audit passes.
- Terrain surface/flow authority audit passes.
- Script and IPC list identical stable keys and exposure state.
- Hiding a connected port is rejected; linking an optional port reveals it.
- Save/reload preserves manual visibility and resolves links by stable key.
- User build verifies Hydraulic, biome-fields and river-network presets before physical pins are deleted.
