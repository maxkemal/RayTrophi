#pragma once

// Per-pass GPU timing for the raster/Realtime viewport frame.
//
// ★★★★★ WHY THIS EXISTS. Measured 2026-09-09: one foliage viewport frame cost
//   ~467 ms, and NOTHING could say what of it was the depth prepass, the shadow
//   atlas, the ray-query dispatch, the main pass or post. The 92 ms the RT
//   shadow handoff saved had to be INFERRED from an A/B on the display loop
//   period, driven one frame per IPC round trip. That works, but it can only
//   ever answer "did this one switch help" -- never "where does the frame go".
//   The same scar is already recorded for the main loop (26.5 ms frame against
//   1.2 ms of measured backend: 95% unmeasured).
//
// ★★★ WHAT THE NUMBERS MEAN, precisely, because a timing that is read wrong is
//   worse than no timing:
//
//   - Marks are written at BOTTOM_OF_PIPE, so mark[i] is "everything recorded
//     before this point has finished". A stage is mark[i+1] - mark[i].
//   - The GPU is free to OVERLAP work across a boundary. A stage number is
//     therefore "time until this boundary retired", not "time this pass would
//     cost alone". They sum to the frame total exactly; they do not attribute
//     an isolated cost. Deleting a 10 ms stage does NOT promise a 10 ms frame.
//   - A stage that did not run this frame reports 0 ms AND ran=false. Those are
//     different states and the caller must not collapse them: "the atlas cost
//     nothing" and "the atlas was handed to the ray pass" look identical in a
//     bare zero.
//
// ★★ Results are read WITHOUT waiting, from a slot old enough to be retired.
//   A timing instrument that stalls the pipeline changes the thing it measures;
//   this one reports "not ready yet" instead, and the caller skips that frame.

#include <array>
#include <cstdint>
#include <vulkan/vulkan.h>

namespace VulkanRT { class VulkanDevice; }

namespace Backend {

// Frame order. Adding a stage means adding it HERE and at its one mark site;
// the pool sizes itself from Count.
enum class RasterStage : std::uint32_t {
    // Frame setup and the GPU culling / LOD compute. Kept separate because
    // "GPU culling silently never enabled" is a fault this repo has already
    // shipped once, and a cost of exactly 0 here is how it looks.
    GpuCull = 0,
    // RT shadow preparation + cascade depth atlas + deep (volume) shadow
    // compute. Measured 2026-09-09: this used to swallow the transmission
    // thickness prepass too and reported 38 ms for an atlas that drew NOTHING.
    ShadowAtlas,
    TransmissionPrep,  // screen-space thickness prepass for refraction
    Sky,
    DepthPrepass,
    RtShadow,          // ray-query screen visibility dispatch
    ScreenGiTrace,
    ScreenGiFilter,
    MainPass,
    VolumeSdf,         // volume raymarch + fluid SDF surface
    // ★★★ Spekuler yansima AYRI olculur, ve bu bir tercih degil sart: kapisi
    //   G-buffer agirligina bagli oldugu icin maliyeti SAHNEYE gore degisir --
    //   parlak yuzeyi olmayan bir sahnede ~sifir, cam/krom dolu bir sahnede
    //   kareyi domine edebilir. MainPass icine gomulu bir maliyet, o farki
    //   "raster yavaslamis" diye okutur.
    Reflection,        // ray-query per-pixel specular reflection
    // ★ Filtre AYRI olculur: yaricapi roughness'la buyudugu icin maliyeti
    //   SAHNEYE gore degisir (aynada sifir tap, kaba yuzeyde 49). Trace ile
    //   toplanmis bir sayi, "filtre bedava" ile "filtre kareyi yedi" arasini
    //   ayirt edemez.
    ReflectionFilter,
    Transmission,
    // ★★★ TAA AYRI olculur. Maliyeti iki parcali (dispatch + tam cozunurluk
    //   kopyasi) ve post'un icine katlanirsa "post pahalilasmis" diye okunur.
    //   Ayrica TAA yakinsarken viewport HER KARE cizer: o donemin bedelini
    //   gormeden "TAA bedava" demek, bu deponun defalarca odedigi hata.
    Taa,
    Post,              // HDR -> LDR display transform + DoF
    Overlay,           // LDR pass: grid, gizmos, hair, particles, edit overlay
    Count
};

const char* rasterStageName(RasterStage stage);

struct RasterGpuFrameTimings {
    // false = no retired slot carried a full set of marks. Every duration below
    // is ABSENT, not zero.
    bool available = false;
    std::array<double, static_cast<std::size_t>(RasterStage::Count)> stageMs{};
    // Did the stage record any work at all this frame. A stage that was skipped
    // has ran=false; a stage that ran and cost nothing measurable has ran=true
    // with ~0 ms. Collapsing the two is how a handoff gets mistaken for a stall.
    std::array<bool, static_cast<std::size_t>(RasterStage::Count)> stageRan{};
    double totalMs = 0.0;
    std::uint64_t frameSerial = 0;
};

class RasterGpuTimers final {
public:
    // Enough rotation that the slot being reset cannot be the one the GPU is
    // still writing: the frame ring admits at most 2 frames in flight, so a
    // 4-deep rotation leaves two frames of margin. Raising the ring's slot
    // count means raising this.
    static constexpr std::uint32_t kSlotCount = 4;

    bool ensure(VulkanRT::VulkanDevice& device);
    void destroy(VulkanRT::VulkanDevice& device);

    // false = this device or queue cannot timestamp. Reported, never faked: a
    // zeroed timing table would be read as "the GPU costs nothing".
    bool supported() const { return m_supported; }
    const char* unsupportedReason() const { return m_reason; }

    // Opens a slot and writes the frame-start mark. Must be called OUTSIDE a
    // render pass (it resets the query range).
    void beginFrame(VkCommandBuffer cmd);
    // Closes `stage`. Safe inside a render pass. Call ONCE PER STAGE, in frame
    // order, EVEN WHEN THE STAGE DID NOT RUN -- pass ran=false for those. A
    // skipped stage still needs its mark, otherwise the query range has a hole
    // and the whole frame becomes uncollectable; and its ~0 ms delta next to
    // ran=false is exactly how the caller tells "handed off" from "free".
    void mark(VkCommandBuffer cmd, RasterStage stage, bool ran);
    void endFrame(VkCommandBuffer cmd);

    // Newest retired slot, or available=false. Never waits.
    RasterGpuFrameTimings collect(VulkanRT::VulkanDevice& device);

private:
    static constexpr std::uint32_t kMarksPerSlot =
        static_cast<std::uint32_t>(RasterStage::Count) + 1u;

    VkQueryPool m_pool = VK_NULL_HANDLE;
    bool m_supported = false;
    bool m_open = false;
    const char* m_reason = "not initialised";
    double m_periodNs = 1.0;
    std::uint32_t m_slot = 0;
    std::uint64_t m_frameSerial = 0;
    // Which marks the frame in each slot actually wrote, and that slot's serial.
    std::array<std::uint32_t, kSlotCount> m_slotMarks{};
    std::array<std::uint64_t, kSlotCount> m_slotSerial{};
    std::array<std::uint32_t, kSlotCount> m_slotStageBits{};
    std::uint32_t m_writtenMarks = 0;
    std::uint32_t m_stageBits = 0;
};

} // namespace Backend
