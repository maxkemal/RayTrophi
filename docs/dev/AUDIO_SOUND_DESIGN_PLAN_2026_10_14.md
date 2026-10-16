# Audio sound design and preview

2026-10-16 continuation: direct-path acoustic preview passed external IPC/native
PCM acceptance; see [the acoustic roadmap](SCENE_ACOUSTICS_ROADMAP.md). The next
implementation slice is specified in
[the background solver plan](SCENE_ACOUSTICS_BACKGROUND_PLAN.md). The subsequent
[GPU source delivery](SCENE_ACOUSTICS_GPU.md) implements fenced direct queries;
user shader/Studio build and live acceptance remain pending. Spectral propagation
and reflections remain future work.

2026-10-15 scope extension: user requested physical scene acoustics and approved
domain-independent object substances first. The source checkpoint is documented in
[OBJECT_SUBSTANCE_BINDINGS.md](OBJECT_SUBSTANCE_BINDINGS.md); propagation stages are
in [SCENE_ACOUSTICS_ROADMAP.md](SCENE_ACOUSTICS_ROADMAP.md). These source additions
do not complete the live propagation/listening gates below.

2026-10-14 — approved direction; first source delivery implemented. Builds and listening
acceptance belong to the user. This document extends the scene audio runtime plan.

## Product direction

Use the existing Audio bottom editor and contextual inspector. Keep the viewport
primary; do not add a permanent shelf. Synthetic and imported sounds share one
sound design workflow: choose, shape, preview, compare, then create an asset.

- Preview in memory before writing a file or adding scene assets. Play/Stop/Loop,
  waveform, level readout, envelope and filter feedback are first priorities.
- Synthetic sound families: tone, impact/resonance, noise/flow. Expose meaningful
  controls and an advanced section for oscillator, envelope, noise and filtering.
- Imported mono/stereo assets: range selection, fades, filters, original/processed
  audition; later add independent pitch/time processing and loop crossfades.
- Reuse miniaudio for playback. Evaluate pinned MIT DaisySP modules for synthesis
  and filtering; Signalsmith Stretch for later independent pitch/time processing.
  Record exact dependency revisions and notices. Do not silently include the
  separate DaisySP-LGPL library. JUCE is an alternative, not the selected framework.

## Saving without typing paths

The normal flow asks for a sound name, not an absolute WAV path. Resolve the folder
from the current project location, falling back to the active asset-library root
for unsaved scenes. Show the destination and allow a folder picker. Use a generated
audio subfolder, safe filenames and automatic numeric suffixes for name collisions.
Do not overwrite an existing sound. Preview needs no output folder or file.

Create/import the WAV only on Save asset. Optional timeline insertion uses the
existing playhead operation. Preserve recipe metadata and existing portable project
packaging. Undo removes the asset/clip, leaving the authored file as documented.
Explicit paths remain available for automation and advanced users.

## Delivery order and shared contracts

1. Shared destination resolver and safe name-based save; in-memory synthesis and
   imported-asset processing; bounded preview voice; visual designer and controls.
2. Richer authored presets and listening refinements, interactive envelope/filter
   curves, spectral analysis and live parameter updates without transport interruption.
3. Independent pitch/time processing, loop crossfade and reusable effect recipes.
4. Physics-driven synthesis using bounded domain/region summaries, then geometry
   acoustics. These remain separate from asset synthesis.

UI, rt.audio and audio.* IPC must call the same AudioService/core implementation.
Validate unknown keys, types, finite bounds and output targets before mutation.
Preview must not mutate assets, enable physics telemetry or participate in timeline
capture. Stop, scene replacement and service destruction must release preview PCM
after the native voice stops. Offline use of saved assets must match preview PCM.

## Acceptance

Static checks: IPC capability/schema parity, Python probe syntax, dependency notices
and project source registration. No builds or Studio launch by Codex.
User build: Studio only for this feature, no additional shaders.
External IPC probe: deterministic generation, no-file preview, asset processing,
invalid requests preserving state, safe filenames/collisions and PCM/save parity.
Manual: hear control changes, original/processed comparison, loop/stop, narrow/high-DPI
layout, Save asset without path entry, timeline insertion, undo and save/reopen.

## Source delivery, 2026-10-14

[Sound designer v2](AUDIO_SOUND_DESIGN.md) implements the first workflow plus
independent pitch/time and loop crossfade: pinned DaisySP/Signalsmith dependencies,
in-memory audition, imported-asset range processing, waveform, draggable envelope,
filter response, sampled spectrum and name-based saving. Unsaved scenes reuse the
active asset-library folder, with runtime-library fallback; saved scenes use their
project folder. UI, Python and IPC share six design service operations.

Static capability audit, binding/source registration checks, enabled dependency
include closure, notice parity, new-source delimiter/whitespace checks and Python
probe syntax passed. These checks do not confirm C++ compilation or audible quality.
Studio build and external probe/manual acceptance remain pending. Further authored
presets, uninterrupted live DSP/background jobs and physics-driven synthesis remain
future stages; they are not marked complete by this asset designer delivery.

## Build checkpoint, 2026-10-14

User confirmed successful Studio compilation after C7692 and M_E compatibility
fixes. New Designer listening/UI/external IPC results have not yet been supplied.
The source/build gate is recorded separately from audible and interaction acceptance.

## User feedback and next authoring direction, 2026-10-14

Synthetic preview was positively tested. The screenshot shows scene timeline
controls, a blank navigation row and three preview-button rows consuming too much
of the bottom editor. The focused layout correction is in source: Designer uses
its own compact preview toolbar, with graphs side by side or in narrow-panel tabs.
The next build must check this layout; other timeline modes keep their transport.

Use familiar audio interactions: logarithmic frequency sliders/knobs, gain in dB,
semitone pitch, draggable ADSR/EQ controls and waveform range/loop handles. Organize
the existing bottom editor/inspector into four workflows without a permanent shelf:

| Workflow | Desired control | Available base / remaining work |
|---|---|---|
| Generate | Oscillator/noise/resonator layers, envelopes, LFO/FM, material-like macro controls | DaisySP components; current integration has oscillator, ADSR and one-pole only. More components/presets must be integrated and auditioned. |
| Edit | Waveform zoom, range selection, trim/split, fades, loop crossfade, independent pitch/time, layering | Existing clips, Designer range/fades and Signalsmith. Interactive source handles and reusable layered design recipes remain to implement. |
| Effects | Ordered bypassable rack, dry/wet and per-effect settings; EQ, dynamics, delay/reverb/modulation | Some DSP components are available in DaisySP; validate each exact module/license before integrating. A rack, routing and expanded effects are not yet implemented. |
| Analyze | Spectrum, spectrogram, peak/RMS/clipping and channel views | Signalsmith Linear FFT/STFT is vendored; current display is a sampled DFT. Full FFT/STFT analysis, spectrogram and interactive inspectors need application work. |

These libraries are processing components, not a ready-made audio editor UI. Pitch/
time/formant controls belong to Signalsmith Stretch; synthesis/effects belong to the
selected DSP modules. Mixing, editing/history, visualization, presets and realtime
scheduling remain RayTrophi service responsibilities. Every new authoring operation
must retain UI/script/IPC parity and deterministic saved/offline behavior.


## V3 visual authoring delivery, 2026-10-14

User approved full visual control. [The implemented v3 contract](AUDIO_DESIGN_TOOLS.md)
now supersedes the remaining-work cells above: five layer types, three layered
presets, nine effects in a reorderable/bypassable rack, source waveform zoom/range
selection, fade/ADSR/EQ/filter/dynamics handles, formant/reverse, full STFT/spectrogram,
channel statistics, Designer undo/redo and saved recipe reuse are implemented in
focused modules with shared UI/script/IPC operations. Source trim is a rendered
selection; timeline split remains the existing clip operation. No destructive file
editor or plugin host is claimed.

The new v3 external acceptance probe is delivered but has not run on a built
application. V2 build/preview feedback remains valid for v2 only. Next: user Studio
build, external probes and manual wide/narrow UI/listening acceptance. Background
rendering/uninterrupted DSP and physics-driven synthesis remain future stages.


2026-10-14 follow-up: user built v3. Both external Designer core/compatibility probes
passed on the open application. Integrated bottom-panel controls, waveform middle-pan/fit
and graph EQ add/remove are now in source; this UI revision needs a new build/manual check.


Current-module visual workspace wiring is now in source: compact module selection,
selected properties, synthesis/effect parameter pads, preset canvas entry and
context-menu rack actions. See AUDIO_DESIGN_TOOLS.md for precise supported controls
and pending build/manual acceptance. Spectral painting/resynthesis, background DSP
and full multitrack editing remain separate core implementation milestones.


## Latest user acceptance and remaining sequence

User rebuilt the shared navigation/audition revision and reported that the issue
did not recur. This is user-reported acceptance, not an independently measured
pass of the new routing regression probe. The subsequent small source edit removes
the inline source Zoom/Pan sliders; wheel/middle-drag navigation, Ctrl modifiers for
source-range view/EQ Q and top Fit remain. This slider-removal edit needs a build.

Continue toward the professional, scene-integrated studio in this order:

1. **Non-destructive multitrack editing:** visual clip placement, trim/split, fades,
   gain/pan envelopes, looping and layer audition/solo; reuse existing clip service
   operations and close gaps in persistent history/project recall.
2. **Background render/analysis:** cancellable jobs, latest-edit ownership, no stale
   audition result, bounded caches and smoother preview without editor-thread stalls.
3. **Spectral editing/resynthesis:** time-frequency region selection and real
   processing, then brushes/masks and reconstruction. Current spectrogram is read-only
   analysis; image editing alone must not be presented as audible editing.
4. **Visual scene-event modulation:** persistent mappings from object/domain events
   to sound parameters, response curves and preview; retain explicit opt-in telemetry.
5. **Project/export and acceptance:** recipe/source portability, channel/routing
   clarity, recording workflow, reference scenes, audio quality and performance checks.

All new operations must share service/core implementation through UI, Python and IPC,
with validation, tests and documentation. Existing capabilities should be composed
first rather than duplicated in a new editor-only model. The current DSP workspace
is a foundation; the milestones above are not marked complete.


Çok katmanlı kurgu aşamasının ayrıntılı iş sırası, canonical servis sınırları ve
kabul ölçütleri [AUDIO_MULTITRACK_AUTHORING_PLAN.md](AUDIO_MULTITRACK_AUTHORING_PLAN.md)
içinde yazıldı. İlk uygulama M0: mevcut clip işlemleriyle ortak kurgu tuvali.
Bağımsız track modeli ve sonraki aşamalar henüz uygulanmış sayılmıyor.


M0 ilk kod entegrasyonu: Edit içinde mevcut clip canvas üzerinden Arrange görünümü,
seçilen klip Properties ve bağlamsal split/duplicate/delete kaynakta yazıldı.
Ayrıntı ve yeni derleme kabul listesi AUDIO_MULTITRACK_AUTHORING_PLAN.md içindedir.


M1 source checkpoint: bağımsız track CRUD/sıralama, kalıcı track_id, ortak canlı/export
mix ve Arrange Tools/Properties bağlantısı yazıldı. Yeni kullanıcı derlemesi,
rt_test_audio_tracks_ipc.py ve proje/undo canlı kabulü bekliyor; bkz. çok katmanlı plan.


M1 kullanıcı build/çalışma kabulü bildirildi. M2 toplu kurgu ve M3 zarf/Arrange
audition-loop ana uygulaması kaynakta tamamlandı; yeni test scripti yazılmadı.
M4 kısa manuel kayıt/yükleme/export kapanışı çok katmanlı planın sonunda listelendi.
Son revizyonun kullanıcı derlemesi ve canlı kabulü bekleniyor.


## 2026-10-15: sıralı devam ve arka plan işleri

Kullanıcı kalan audio yol haritasının sırayla sürdürülmesini istedi. Açık boş sahnede
mevcut binary'nin track/export probu ve dar M2/M3 batch/curve/Arrange IPC turu geçti;
test sonunda assets/clips/tracks yeniden sıfırdı. Manuel undo/redo, proje round-trip,
işitsel kalite ve performans kabulü bundan ayrı bekliyor.

Sıradaki arka plan aşamasının ilk kaynak teslimi yazıldı:
[AUDIO_BACKGROUND_JOBS.md](AUDIO_BACKGROUND_JOBS.md). Designer üretimi ve Analyze
STFT'si bounded worker üzerinden çalışır; UI/Python/IPC aynı start/status/cancel/apply
servisini kullanır. En yeni iş sahipliği, revision/asset/dosya değişimi denetimi ve
iptal bulunur. Yeni kodun kullanıcı Studio derlemesi ve yeni dış probu bekleniyor.

Bu teslim bütün arka plan aşamasını tamamlamaz: Arrange miks üretimi, source waveform,
history/reuse ve dosya save/import yolları hâlâ çağıran thread'dedir. Sonraki sıra:
bu yollar ve kesintisiz DSP/performance kabulü; gerçek spectral processing; görsel
scene-event mapping; taşınabilir proje/export ve kalan runtime kabul kapıları.

## 2026-10-15: testleri sona erteleme ve worker yollarının genişletilmesi

Kullanıcı yeni test scriptleri ve canlı test turlarını yapısal işlerin sonuna
erteledi. Kaynak geliştirme ve statik sözleşme denetimleri sürer; yeni teslimler
derlenmiş veya canlı kabul edilmiş sayılmaz.

Arrange decode/mix, Designer source waveform decode/envelope ve tarif undo/redo/reuse
yeniden üretimi için arka plan yolları kaynakta bağlandı. UI, Python ve IPC aynı
`background=true` servis seçeneklerini kullanır; eski synchronous sözleşmeler
korunur. İptal/en yeni sonuç sahipliği, dosya/snapshot güncelliği ve geçmişin yalnız
başarılı apply ile değiştirilmesi uygulanır. Ayrıntı AUDIO_BACKGROUND_JOBS.md içinde.

Bu revizyon bütün planın kapanışı değildir: save/import dosya yolları, kesintisiz
DSP/crossfade, gerçek spektral işleme, görsel scene-event mapping ve proje/export
kapanışı hâlâ açık kalır. Son test turu bu işlerin ardından yapılacak.

### Gerçek spektral bölge işleme, aynı kaynak turu

`spectral_region` efekt çekirdeği ve Analyze üzerinde zaman/frekans seçimi kaynakta
eklendi. Dikdörtgen/elips maskesi, gain, feather, kanal ve dry/wet mevcut tarif/rack
üzerinden saklanır. Pinned FFT ile fazı koruyan magnitude değişimi ve padded
overlap-add gerçek PCM üretir; UI/script/IPC aynı çekirdeği kullanır. Bu artık yalnız
görsel spektrogram düzenlemesi değildir. Ayrıntı AUDIO_SPECTRAL_EDITING.md içinde.

Serbest fırça/büyük keyfi maskeler, realtime DSP/crossfade, scene-event mapping,
dosya save/import worker yolları ve proje/export kapanışı açık. Yeni kaynak henüz
derlenmedi; test scriptleri ve işitsel kabul kullanıcının istediği son aşamaya kaldı.
