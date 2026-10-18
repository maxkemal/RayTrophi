# Saved acoustic settings and inspector–mixer connection

2026-10-18 source delivery. No build, shader compilation, application launch or
live test was performed for this change. The previous 20-case acceptance predates
these source changes and does not validate persistence/inheritance or the new UI.

## Scene and source ownership

The audio scene snapshot now contains `acoustics`, a canonical config with
`enabled=false`, `meters_per_unit=1`, `boundary_mode="object_barrier"`,
`spectral=true`, `reflections=false`, `diffuse=false` defaults. The same six fields
retain AcousticPreview's existing boolean/scale/model validation. Unknown fields
reject. Scene update merges a config patch before validation.

Every normalized emitter has `acoustics={mode,config}`. Mode is `scene` (default),
`custom` or `off`. Custom config has the same validated six fields. `update` replaces
this policy object; include its config when retaining an existing custom recipe.
The UI preserves it when switching off, and seeds custom settings from the effective
scene config when switching from scene inheritance. Reset changes only the mode
back to scene. New/duplicated/restored sources use the same normalization path.
Missing fields in older scenes default to disabled scene acoustics and scene mode.

Source/scene configs live in AudioService's existing saved snapshot and therefore
follow project serialization and authoring undo. Raw query results, computed paths,
preview revisions and dry/processed bypass are held only in AcousticPreview and
are rebuilt after restore. No computed results or A/B state are serialized.

Scene defaults apply to existing inheriting sources and subsequently added ones.
Custom/off sources remain exceptions until explicitly reset. Non-spatial sources
are excluded from geometry acoustics; this does not turn off ordinary playback.
Inherited diffuse scattering is excluded for directional sources while supported
direct/specular processing remains active. The UI explains that exclusion. Explicit
enabled custom diffuse on a directional spatial source rejects.

The current acoustic query budget remains 32 enabled spatial sources. A change
requiring more rejects atomically; it does not silently choose a subset. Candidate
scene and runtime entries are staged before publication. Source create preflights
before allocating its ID, update before touching a native voice, and restore
before replacing live data. Unchanged configs retain runtime estimates/revisions
and session bypass; changed configs receive fresh pending queries.

## Shared Python / IPC operations

Python uses `rt.audio.*`; external IPC uses `audio.*` with the same parameters.

| Operation | Contract |
|---|---|
| acoustics_get | No params; scene config and per-source policy/effective config/eligibility |
| acoustics_update | `config` object patch of saved scene defaults; preserves exceptions |
| acoustics_reset_sources | `scope=selected/group/scene`, optional `ids`; return targets to scene inheritance |
| create / update | Optional `acoustics` source policy object |
| acoustics_preview | Existing id + enabled/config fields; now authors a saved custom policy |
| acoustics_batch configure | Existing scope/ids/config; now authors custom policies atomically |
| acoustics_batch dry/processed | Existing scope/ids; session bypass only, no authored data change |

Get is Read. Scene update/reset and acoustic preview/batch are SceneWrite;
create/update keep their existing file capability requirements. Invalid requests
keep the established Python ValueError / IPC audio_error translation. The legacy
preview APIs still require explicit boolean enabled. Turning them off creates a
disabled custom exception; it does not erase authored state and resume inheritance.

```python
rt.audio.acoustics_update(config={"enabled": True, "reflections": True})
rt.audio.update(id=source_id, acoustics={"mode": "off"})
rt.audio.acoustics_reset_sources(scope="scene")
rt.audio.workspace_update(mixer_bus="Effects")
```

## UI and mixer

Audio > Scene exposes saved scene defaults and an explicit reset-all-exceptions
operation. Source > Acoustics chooses Use scene settings / Custom settings / Off.
Inherited controls are read-only and identify where their scene defaults are edited.
No nested fixed-height acoustic area is reintroduced.

Source > Sound shows the assigned mixer channel's level in dB, mute and real peak
meter. These are the same mixer_update/mixer_get values as the bottom mixer, not
copied settings. The text explicitly says all sounds routed to that channel are
affected. Channel meter readings are aggregated and before preview output mute;
they are not a selected-source or speaker-output meter. Asset audition retains its
existing mixer-bypass behavior.

Show channel in Mixer sets transient workspace `mixer_bus` (validated against the
canonical bus list), closes Designer view, and opens the existing bottom Mixer.
That strip is labeled Selected, uses the selected accent and scrolls into view on
focus change. `workspace_get/update(mixer_bus=...)` exposes the same selection to
Python/IPC; the existing editor API controls bottom-editor navigation. No second
mixer, per-source bus copy or persistence of this presentation focus is introduced.

## Next user build / acceptance

Existing `rt_test_acoustic_playback_ipc.py` is extended, not run: new source
inheritance, scene updates preserving custom/off exceptions, invalid edits retaining
authored snapshots, reset-to-scene, workspace mixer focus and A/B preserving saved
settings. Its preexisting concurrent/PCM/acoustic checks remain.

After the user build, run it externally on a disposable empty scene. Also check:

- Save a scene with scene defaults, a custom source and an off source; reopen it.
  Configs/modes must return, runtime entries start pending, and previous Dry bypass
  must not return. Undo/redo scene defaults and source policy changes.
- Add a new spatial source after enabling scene acoustics; confirm it inherits.
  Non-spatial audio is excluded, and directional inherited diffuse is explained.
- Try the 32-source acoustic limit; rejected enable/create/update/restore must
  preserve authored state and existing runtime entries.
- In Sound, change channel level/mute and verify the bottom mixer mirrors it;
  change the bottom mixer and verify the inspector. Show channel should highlight
  and reveal the correct bus, including after source routing changes.

Static validation covers lexical delimiter balance (Windows branch selected for
the existing platform-specific loader), VS XML registrations, Python AST, source
size and whitespace. It is not C++ compilation, round-trip or live UI acceptance.
