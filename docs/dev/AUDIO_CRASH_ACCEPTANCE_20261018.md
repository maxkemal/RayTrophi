# Audio acceptance crash, 2026-10-18

## Rebuilt application follow-up

Final rebuilt follow-up: the full external acceptance completed successfully
on 2026-10-18 with 20 cases and all 28 malformed requests rejected without
authored mutation. Group mix ID permutation, doubled gain, exact sample crop,
unequal history lengths, invalid request job preservation, group stale rejection,
single-source bake, native replay and sampled history passed. This supersedes
the group-pending status below. The current full report is
`audio_physical_acceptance_20261018.json` with `passed=true`.
One cleanup warning remains: the test substance is referenced by undo history;
its geometry and all test audio sources/assets were removed. No build, launch
or scene save was performed by the agent.

The user rebuilt and reopened the disposable scene without the cube. The full
external acceptance run now safely rejected missing/null/numeric/empty IDs and
completed 17 cases: reactive acoustic pose/occlusion/reflections/lifecycle,
seven continuous generator models and four explicit-impact resonators. The
occluded/direct RMS ratio was 0.2000037093 against an expected 0.2.

It then rejected group bake with `audio params must be an object`, without a
crash. The internal `service.invoke("list", {})` constructed JSON null. Changed
this call to `json::object()`; that source fix requires another user build.
The group mix/permutation/crop/unequal-length checks have not yet run.

To reach the remaining checks without repeating accepted PCM cases, the same
script was narrowed in memory for a separate external continuation. This run
passed single-source deterministic bake/stale rejection/stored history, seeked
native replay, illegal injection/pitch rejection, retirement and sampled-history
store/bake. Repeated PCM digests were both `788a5b0e805c7c41`. Reports:
`audio_physical_acceptance_20261018.json` (partial, failed at group start) and
`audio_physical_continuation_20261018.json` (passed within its explicit scope).

The full run removed its source/geometry resources; its test substance remains
referenced by undo history and the substance API correctly refused deletion.
The narrowed run's substance cleanup reports unknown substance because that
fixture was deliberately skipped. No project save, build or app launch occurred.
This supersedes the pending statuses below only for the stated live checks.

The user built Studio and authorized the open scene as a disposable test area.
The external `rt_test_reactive_acoustics_ipc.py` run stopped before its first
acceptance case completed. The scene's Default_Cube was removed before the run.
No application build or launch was performed by the agent.

Evidence: `x64/Release/StartupCrashSignal.dmp`, `StartupCrash.log` and the latest
entry in `StartupCrashStack.txt`, timestamp 12:57:52. The dump was read with the
installed Windows SDK cdb and the matching local PDB. Its synthetic exception
code is e0000016; the application log records signal 22 (SIGABRT).

The symbolized stack is `ucrtbase!abort` -> nlohmann JSON `at()` ->
`audio::AudioService::invoke` (AudioService.cpp:724 in that build) -> audio author
-> IPC dispatch -> main-thread queue. The return address immediately follows
the source_runtime parameter lookup. The test intentionally sends `{}` to that
operation immediately before its first signal-drive call. The subsequent pipe
error 232 is a consequence of the process exiting, not evidence that signal_set
caused the fault. No UTF-8 error appears in the captured stack.

The linked JSON accessor takes an abort path on missing keys in this build.
The precise compilation/ODR origin of that behavior has not been established.
Do not assume a catch around JSON access will safely reject malformed input.

Source correction: validate source_runtime id presence, string type and nonempty
content with the service's explicit invalid_argument check before JSON access.
The existing negative fixture now covers omitted/null/numeric/empty IDs as well
as unknown IDs and unexpected parameters. Source guard ordering, Python AST and
diff whitespace checks passed; this correction has not been built or live-tested.

Next gate: user rebuild, reopen the disposable scene, then rerun the same external
acceptance script. Current report `audio_physical_acceptance_20261018.json` is
failed with zero accepted cases. Its cleanup calls could not run after the crash;
the process exit released runtime resources and the scene was not saved. Full
procedural PCM, bake/group/range/replay/history acceptance remains pending.
