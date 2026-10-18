# Audio request envelope validation

Source checkpoint, 2026-10-18. No build or additional live test performed.
Subsequent user-built live acceptance passed on 2026-10-18: the full external
reactive-acoustics fixture completed 20 cases, including all 28 malformed
requests with unchanged authored audio state. Report:
`audio_physical_acceptance_20261018.json`. This supersedes the source-only and
next-build status below for the covered paths; it does not prove all nested
schemas or JSON compilation settings are safe.
This follows the dump-confirmed source_runtime missing-ID abort documented in
AUDIO_CRASH_ACCEPTANCE_20261018.md. The linked JSON accessor can abort on invalid
access in the current application build; catching an exception is insufficient.

AudioRequestValidation checks common top-level string, boolean, finite numeric,
integer, object, array and three-component vector fields before typed access.
IDs are nonempty strings; source/asset/clip/track operations that need an ID
reject omission. Physical impact requires sequence/energy; physical bake needs
path plus source selection; transport/mixer/signal and selected other entry
points check their required fields. Existing operation-specific range, unknown
field, existence and state validation remains authoritative.

AudioAuthoring calls the same validator before reaction-domain and parent
inspection, after its existing default asset-root resolution. AudioService
calls it before dispatch, covering direct UI/core calls. Emitter normalization
also invokes it for restored records. Reaction validation explicitly checks
string/boolean fields and numeric types before conversion, including callers
outside the request dispatcher. No duplicate reaction business logic is added.

Both Python and IPC already use AudioAuthoring and the shared service. Bad
requests raise invalid_argument and retain their existing ValueError/audio_error
transport semantics; they must not mutate authored scene state. New modules
are registered in the VS project/filter and discovered by the CMake Scene glob.

This is envelope and emitter/reaction hardening, not a proof that every nested
designer/effect/thunder/clip patch schema or every persistence input is safe.
The wider mixed-exception JSON compilation/ODR audit remains separate.

The existing external reactive-acoustics fixture now includes 28 malformed
requests and compares authored audio snapshots after every rejection. This
expanded block requires the next user build. Do not run it on the older binary:
some malformed inputs intentionally exercise the newly guarded paths.
The already-passed 17 acoustic/generator cases and isolated bake/replay/history
continuation are unchanged evidence, not acceptance of this source update.
