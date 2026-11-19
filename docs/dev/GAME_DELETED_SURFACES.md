# Deleted game surface references

`Paint::reconcileGameDesign` removes wet paint registrations whose canonical flat
node is missing or pending deletion. It also removes managed PaintCollider entries
bound to missing objects, and colliders bound to the surfaces removed in that
reconciliation, across the scene's particle runtimes. A replacement wall is never
chosen by name similarity. Other valid surfaces and their paint settings remain.

The shared simulation clock reconciles before simulation/paint work. Existing
wet status, demo and design API calls also reconcile, so UI, Python and IPC use
the same core state even between ticks. Pending project loads and simulation
bakes are excluded; API entry points skip mutation during final render jobs.
There is no new IPC method, shader or public data layout.

The Game Design panel clears the obsolete missing-paint-surface error once the
shared service has removed that object's registration. Save reads the live list,
so the removed surface is absent from subsequent `game_design` serialization.
Cleanup removes the registration; undoing mesh deletion alone does not recreate
its former paint registration. The restored mesh can be added as a surface again.

## Verification

No C++ build or application launch by Codex. User: incremental Studio and Player
build (GameDesign header/core integration); shader compilation not needed.

After build, externally run with a disposable Studio game scene containing the
four BalloonPaint droplet systems:

```powershell
python scripts/test/rt_game_deleted_surface_ipc.py
```

The fixture adds a temporary cube, registers paint and an OBB collider, deletes
the cube, and checks runtime surfaces, demo colliders and serialized design all
exclude it. It cleans its own registrations and never saves the project.
Its syntax was checked; live execution awaits the rebuilt application.

UI acceptance: delete a registered wall; verify its row disappears and Reconfigure
does not search for it. Add/register a new wall and verify its own settings are
used. Save/reopen and check the deleted name does not return. A real SDF cooking
error for an existing wall remains an error; this change does not suppress it.
