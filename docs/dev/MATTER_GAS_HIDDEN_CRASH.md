# Matter gas Fog -> Hidden crash

2026-11-06: user reported an application crash when Output gas/vapor was changed
from Fog to Hidden. The current `x64/Release/StartupCrash.log` records three fatal
exceptions at 23:49:08, 23:49:33 and 23:50:14, all with code `0xC0000094`
(integer divide by zero), at the same address. The later terminate backtrace is
shutdown fallout, not the original fault's call stack. The available
`StartupCrashSignal.dmp` is dated 2026-10-22, so it is not evidence for this event.

Source path: hidden gas makes `gas_enabled=false`, `gas_streams=0` and
`gas_cells=0` in `MatterFogFields.cpp`. Budget accounting nevertheless calls
`charge(0, 0, budget, total)`. The old overflow check evaluates
`(budget - total) / bytes`, dividing by zero even though the requested allocation
is empty. This deterministic source path agrees with the logged exception type;
the log does not contain a symbolized original-fault frame identifying the line.

Fix: preserve rejection when the existing total exceeds budget, then accept
zero-count or zero-byte charges without division or changing the total. Nonempty
charges retain the overflow-safe division and existing budget rejection.
No rendering buffer lifetime/transport changes are needed for this defect.

User build/acceptance (no C++ test files; agent did not compile or launch app):

- Rebuild C++; this fix itself needs no shader compilation.
- In the original scene, switch gas/vapor Fog -> Hidden -> Fog several times,
  paused and during ordinary playback. It must survive; the gas stream must
  disappear/reappear while carrier SDF/splat/fog remains correctly routed.
- Also check a domain with all outputs hidden (zero fields), and one with only
  carrier Fog and gas Hidden. Neither empty-gas case may divide by zero.
- The same action is available through external named-pipe IPC:
  `fluid.set_phase_output(domain=<name>, gas="hidden")`, then gas="fog".
  Preserve/restore the original phase-output patch; do not reset/seed the scene
  merely to test output visibility. Use `fluid.get_phase_output` to read it back.
- After rebuilding, the old crash log remains historical evidence; compare new
  entries rather than treating its old lines as another failure.

Source/whitespace checks passed. Live post-build verification is pending.

User subsequently rebuilt and confirmed the crash is fixed.
