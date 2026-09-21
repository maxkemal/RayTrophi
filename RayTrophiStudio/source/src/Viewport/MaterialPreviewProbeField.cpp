#include "Backend/VulkanBackend.h"
#include "RayFusion/ProbeField.h"
#include "globals.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace Backend {
namespace {

// DEFAULT window, not a build constant any more. The shape and placement of the
// field are runtime values (rayfusion.set_probe_grid): coverage is the input
// every RayFusion image question so far has been confounded by -- a 12x6x12 box
// nailed to the world origin, with a measured hit_fraction of 0.105 because most
// of its probes sit outdoors -- and a parameter that needs a rebuild to change
// cannot be A/B'd against the thing it is blamed for.
//
// The defaults reproduce the previous field byte for byte: nothing gets slower,
// or different, until a caller asks for a different window.
constexpr uint32_t kDefaultCountX = 4u;
constexpr uint32_t kDefaultCountY = 2u;
constexpr uint32_t kDefaultCountZ = 4u;
constexpr float    kDefaultSpacing = 3.0f;
constexpr int64_t  kDefaultMinCell[3] = {-2, -1, -2};
// Ray cost per frame is bounded by the BUDGET (probes per update), not by the
// slot count, so a larger window costs convergence latency and buffer bytes --
// not frame time. The buffer is sized for the ceiling once and never resized.
constexpr uint32_t kMaxProbeSlots = RayFusion::kMaxProbeSlots;

// Sky-bake producer only: every probe sees all of the sky, so a small distance
// here would invent a visibility measurement that was never taken. The TRACED
// producer does not use this -- it reports what the rays actually found.
constexpr float kUnoccludedDistance = 1.0e4f;

// Bir probe'un ışınlarının bu kadarı ARKA yüze çarptıysa probe geometrinin
// içinde doğmuştur. O probe'un doğru cevabı "siyah" değil "olculemedi"dir:
// paket alfa 0 ile yayinlanir, slot GECERLI sayilir (yoksa her kare yeniden
// izlenirdi) ve tuketici kendi global okumasina geri duser. Sessizce siyah
// yazmak, olculmemis bir degeri olcum gibi gostermek olurdu.
constexpr float kInsideGeometryBackfaceFraction = 0.35f;
// ★★★★★ VE ARKA YUZ ORANI TEK BASINA YETMEZ. Olculdu 2026-09-15: kapali bir
//   yatak odasinda isabetlerin %84,5'i ARKA yuz -- duvarlarin normalleri disa
//   baktigi icin. Yani odanin TAM ORTASINDAKI probe da bu esigi asiyordu ve
//   "geometri icinde dogdu" diye ATILIYORDU. Kullanicinin gordugu: "oda icinde
//   problar zaten kirmiziya donmus halde."
//
// ★★★★ Sezgi "cok arka yuz => katinin icindeyim" diyor. Bu bir KATI icin dogru,
//   ama disa bakan duvarlardan kurulu KAPALI BIR ODA icin de birebir ayni
//   gorunur. Iki durumu arka yuz orani AYIRT EDEMEZ.
//
// ★★★ Ayirt eden sey MESAFE: duvara gomulu bir probe'un etrafindaki geometri
//   santimetrelerde, odadaki bir probe'unki metrelerde. Olcek hucre boyutudur
//   (`spacing`), cunku probe alani zaten o olcekte kurulur.
// ★★ Shader degisikligi GEREKMEDI: `distance[0]` zaten yone gore kosinus
//   agirlikli ortalama carpma mesafesi olarak yaziliyor.
constexpr float kInsideGeometryDistanceFraction = 0.25f;

// ★★★★★ IZGARA YERLESIMI OTOMATIKTIR, ve bunun gerekcesi bir OLCUM.
//   2026-09-15, kapali bir yatak odasi: `hit_fraction` %3,2 -- probe
//   isinlarinin %96,8'i hicbir seye carpmiyordu -- ve `mean_hit_distance`
//   194 m. Yani 1024 slotluk butcenin 256'si BOS HAVAYA dizilmisti.
//   Kullanicinin gordugu belirti bu degildi: "duvarin yarisi mavi yarisi
//   dogal renginde" idi. Izgaranin disinda kalan piksel, probe okumasi da
//   gorunurluk tahmini de sustugu icin ENGELSIZ gokyuzune geri dusuyordu.
//
// ★★★★ Iki emniyet AYNI YERDE susuyordu (`rfSampleProbeField` false,
//   `rfSpecularSkyVisibility` 1.0), yani ariza korelasyonlu. Bu yuzden gecis
//   bir gradyan degil KESKIN BIR CIZGI: tam olarak bildirilen belirti.
//
// ★★★ Elle ayarlanabilir birakmak cozum degildi: izgarayi her sahne icin
//   elle oturtmak, zincirin ortasina manuel bir adim koymak demek.

// Bir eksende en fazla kac hucre. Izgaranin kendi API'siyle ayni sinir.
constexpr uint32_t kAutoFitMaxCount = 128u;
// Kamera-yerel rejimin kapsadigi dunya hacmi (metre). Butun butce BUNUN
// icine harcanir; sahne buyukse tamamini kaplamak zaten imkansiz, ve kaba
// bir izgarayla her yeri kaplamak, ince bir izgarayla yakini kaplamaktan
// daha kotudur -- GI yerel bir olaydir.
// ★★★ 18x9x18 ilk denemeydi ve OLCUM onu eledi: kullanicinin elle kurdugu
//   kutu (20x12x28) `hit_fraction`i %3,2'den %59,2'ye cikardi, benim
//   varsayilanim ise odayi kapsamiyordu. Buyutmenin bedelini artik 4096'lik
//   tavan odyor -- eskiden ayni hacim ya kaba ya eksik olmak zorundaydi.
constexpr float kAutoFitLocalExtent[3] = {24.0f, 14.0f, 24.0f};
// Sahne rejiminde kutunun her yanina eklenen pay. Duvarin DISINDA da probe
// olmali, yoksa duvar yuzeyi izgara sinirina dusler ve tam kacindigimiz
// kenar dikisi geri gelir.
constexpr float kAutoFitPadFraction = 0.12f;

// Verilen dunya boyutunu butceye sigan EN INCE aralikla ortmeye calisir.
// Basarisizlik diye bir hali yok: aralik buyutulerek her zaman sigdirilir --
// ama CAGIRAN sonucun ne kadar kabalastigini gorebilsin diye aralik geri
// dondurulur, cunku "sigdi" ile "kullanilabilir yogunlukta sigdi" ayni sey
// degildir.
float autoFitSpacingFor(const float extent[3], uint32_t budget,
                        std::array<uint32_t, 3>& outCounts) {
    // Hacim/butce kok tahmini dogru buyukluk mertebesini tek adimda verir;
    // asagidaki dongu yalnizca yuvarlama payini kapatir.
    double volume = 1.0;
    for (int axis = 0; axis < 3; ++axis)
        volume *= static_cast<double>((std::max)(extent[axis], 0.01f));
    float spacing = static_cast<float>(std::cbrt(volume / (std::max)(1u, budget)));
    if (!std::isfinite(spacing) || spacing <= 0.0f) spacing = kDefaultSpacing;
    for (int axis = 0; axis < 3; ++axis)
        spacing = (std::max)(spacing, extent[axis] / static_cast<float>(kAutoFitMaxCount));
    // 0,05 adimina yukari yuvarlama: kamera kimildadikca aralik 1,4998 ile
    // 1,5001 arasinda gidip gelirse her kare RECONFIGURE olur ve reconfigure
    // butun alani gecersiz kilar -- yani alan hicbir zaman yakinsamaz.
    auto quantize = [](float v) {
        return (std::max)(0.05f, std::ceil(v * 20.0f) / 20.0f);
    };
    spacing = quantize(spacing);
    for (int guard = 0; guard < 128; ++guard) {
        uint64_t slots = 1u;
        for (int axis = 0; axis < 3; ++axis) {
            // +1: hucre sayisi degil PROBE sayisi lazim. Tam bir hucreye
            // oturan bir kutunun iki YANINDA da probe olmali, yoksa
            // `rfSampleProbeField`in 8 komsusu kenarda hic dolmaz.
            const uint32_t n = static_cast<uint32_t>(
                std::ceil(extent[axis] / spacing)) + 1u;
            outCounts[axis] = std::clamp(n, 1u, kAutoFitMaxCount);
            slots *= outCounts[axis];
        }
        if (slots <= budget) break;
        spacing = quantize(spacing * 1.08f);
    }
    return spacing;
}


const char* qualityPresetName() {
    switch (::render_settings.raster_viewport_quality_preset) {
        case ::RasterViewportQualityPreset::Performance: return "performance";
        case ::RasterViewportQualityPreset::Quality:     return "quality";
        case ::RasterViewportQualityPreset::Full:        return "full";
        default:                                         return "balanced";
    }
}

// Must match rfOctDecode in probe_field.glsl exactly. If the direction that
// writes a texel and the direction that reads it drift apart, the symptom is
// "the ambient is slightly wrong" -- which nobody reports as a bug.
void octDecode(float ex, float ey, float out[3]) {
    float x = ex, y = ey;
    const float z = 1.0f - std::abs(ex) - std::abs(ey);
    if (z < 0.0f) {
        const float sx = x >= 0.0f ? 1.0f : -1.0f;
        const float sy = y >= 0.0f ? 1.0f : -1.0f;
        const float nx = (1.0f - std::abs(y)) * sx;
        const float ny = (1.0f - std::abs(x)) * sy;
        x = nx; y = ny;
    }
    const float len = std::sqrt(x * x + y * y + z * z);
    const float inv = len > 1e-8f ? 1.0f / len : 0.0f;
    out[0] = x * inv; out[1] = y * inv; out[2] = z * inv;
}

// Same mapping as directionToUv in material_preview_ibl.comp. Bilinear, because
// 64 octahedral texels are being pulled out of a 64x32 irradiance map.
void sampleEquirect(const std::vector<float>& rgba, uint32_t width, uint32_t height,
                    const float d[3], float out[3]) {
    out[0] = out[1] = out[2] = 0.0f;
    if (rgba.empty() || width == 0u || height == 0u) return;
    constexpr float kPi = 3.14159265358979323846f;
    float u = std::atan2(d[2], d[0]) / (2.0f * kPi) + 0.5f;
    u -= std::floor(u);
    const float v = std::acos(std::max(-1.0f, std::min(1.0f, d[1]))) / kPi;

    const float fx = u * static_cast<float>(width) - 0.5f;
    const float fy = std::max(0.0f, std::min(static_cast<float>(height) - 1.0f,
                                             v * static_cast<float>(height) - 0.5f));
    const int x0 = static_cast<int>(std::floor(fx));
    const int y0 = static_cast<int>(std::floor(fy));
    const float tx = fx - static_cast<float>(x0);
    const float ty = fy - static_cast<float>(y0);
    const auto wrapX = [&](int x) {
        const int w = static_cast<int>(width);
        return ((x % w) + w) % w;
    };
    const auto clampY = [&](int y) {
        return std::max(0, std::min(static_cast<int>(height) - 1, y));
    };
    const int xs[2] = {wrapX(x0), wrapX(x0 + 1)};
    const int ys[2] = {clampY(y0), clampY(y0 + 1)};
    const float wx[2] = {1.0f - tx, tx};
    const float wy[2] = {1.0f - ty, ty};
    for (int j = 0; j < 2; ++j)
        for (int i = 0; i < 2; ++i) {
            const size_t base =
                (static_cast<size_t>(ys[j]) * width + static_cast<size_t>(xs[i])) * 4u;
            if (base + 2u >= rgba.size()) continue;
            const float w = wx[i] * wy[j];
            out[0] += rgba[base + 0] * w;
            out[1] += rgba[base + 1] * w;
            out[2] += rgba[base + 2] * w;
        }
}

uint32_t slotsFor(const std::array<uint32_t, 3>& counts) {
    return counts[0] * counts[1] * counts[2];
}

// Camera/centre placement puts the given world point in the MIDDLE of the
// window. Both the follow-camera path and an explicit `center` resolve through
// this one function; two copies of it would put the two placements half a cell
// apart and the symptom would be "following moved the light slightly".
RayFusion::Cell windowMinimumFor(const float centre[3], float spacing,
                                 const std::array<uint32_t, 3>& counts) {
    RayFusion::Cell minimum{};
    for (int axis = 0; axis < 3; ++axis) {
        const int64_t cell = static_cast<int64_t>(std::floor(centre[axis] / spacing));
        minimum[axis] = cell - static_cast<int64_t>(counts[axis] / 2);
    }
    return minimum;
}

struct alignas(16) ProbeGridParams {
    int32_t counts[4];   // xyz cell counts, w = published slot count
    int32_t minimum[4];  // xyz lowest cell of the window, w = field active
    float   spacing[4];  // x = cell size in world units, yzw reserved
};
static_assert(sizeof(ProbeGridParams) == 48u, "probe_field.glsl grid ABI");

} // namespace

class MaterialPreviewProbeResources {
public:
    RayFusion::ProbeField field;
    VulkanRT::BufferHandle texels;      // slot-major array of ProbeTexel
    VulkanRT::BufferHandle gridParams;
    VkDescriptorSet boundSet = VK_NULL_HANDLE;
    // Producer side: the CPU copy of the last bake and the signature naming it.
    std::vector<float> sourceIrradiance; // RGBA32F equirect
    uint32_t sourceWidth = 0;
    uint32_t sourceHeight = 0;
    uint64_t sourceSignature = 0;
    uint64_t appliedRevision = 0;
    bool configured = false;
    bool ready = false;
    bool uploaded = false;
    // Producer selection. This is an A/B MEASUREMENT lever, not a permanent
    // second code path: 1b has to be comparable against 1a on the same scene,
    // and a fix (or a producer) that cannot be switched off also destroys the
    // measurement that would have judged it.
    bool preferTraced = true;
    bool overlay = false, overlayReady = false, followCamera = false;
    // Otomatik yerlesim VARSAYILAN ACIK. Kapatilabilir olmasi sart: bir
    // duzeltmeyi kapatamamak, onu yargilayacak A/B olcumunu de yok eder.
    bool autoFit = true;
    std::string autoFitMode = "off";    // NE KOSTU: scene | camera_local | off
    std::string autoFitReason;
    uint64_t autoFitSignature = 0;      // yeniden oturtmayi tetikleyen girdiler
    bool pump = false;
    // REQUESTED window. The field itself holds the APPLIED one, and status
    // reports that one -- these two differ only between a set_probe_grid call
    // and the next service pass. `reconfigure` means the slot array itself has
    // to be rebuilt (counts or spacing); a pure placement change scrolls.
    std::array<uint32_t, 3> counts{kDefaultCountX, kDefaultCountY, kDefaultCountZ};
    float spacing = kDefaultSpacing;
    RayFusion::Cell minimum{kDefaultMinCell[0], kDefaultMinCell[1], kDefaultMinCell[2]};
    bool reconfigure = false;
    uint32_t overlayMarkers = 0;
    std::string overlayReason;
    std::string activeProducer = "none";   // what actually ran, not what was asked
    bool activeBounce = false;
    std::string producerReason;            // why the request was not honoured
    uint32_t rejectedInside = 0;           // probes born inside geometry
    // ★★★ Mesafe kapisinin KURTARDIGI probe sayisi. Bu sayac olmadan duzeltme
    //   dogrulanamaz: "kirmizi probe kalmadi" hem "kapi calisti" hem "hic probe
    //   yayinlanmadi" demek olabilir.
    uint32_t backfaceEnclosed = 0;         // cok arka yuz AMA katinin icinde degil
    float lastHitFraction = 0.0f;          // rays that hit anything, 0..1
    float lastMeanHitDistance = 0.0f;      // yonler uzerinden ortalama, dunya birimi
    double lastTraceMs = 0.0;
    uint64_t tracedPublishes = 0;
};

bool VulkanBackendAdapter::ensureMaterialPreviewProbeResources() {
    if (!m_device || !m_device->isInitialized()) return false;
    auto state = m_materialPreviewProbes;
    if (state && state->ready) return true;
    if (!state) {
        state = std::make_shared<MaterialPreviewProbeResources>();
        m_materialPreviewProbes = state;
    }

    // Allocated for the CEILING, once. Reallocating on a grid change would put
    // a freed buffer under an in-flight frame, and the descriptor set written
    // at bind time points at this handle -- a new handle would need a rebind
    // the shader has no way to ask for.
    VulkanRT::BufferCreateInfo texelInfo;
    texelInfo.size = static_cast<uint64_t>(kMaxProbeSlots) * RayFusion::kProbeTexels *
                     sizeof(RayFusion::ProbeTexel);
    texelInfo.usage = VulkanRT::BufferUsage::STORAGE | VulkanRT::BufferUsage::TRANSFER_DST;
    texelInfo.location = VulkanRT::MemoryLocation::CPU_TO_GPU;
    state->texels = m_device->createBuffer(texelInfo);

    VulkanRT::BufferCreateInfo gridInfo;
    gridInfo.size = sizeof(ProbeGridParams);
    gridInfo.usage = VulkanRT::BufferUsage::STORAGE | VulkanRT::BufferUsage::TRANSFER_DST;
    gridInfo.location = VulkanRT::MemoryLocation::CPU_TO_GPU;
    state->gridParams = m_device->createBuffer(gridInfo);

    if (!state->texels.buffer || !state->gridParams.buffer) {
        destroyMaterialPreviewProbeResources();
        return false;
    }

    // Starts INACTIVE on purpose. The shader reads this buffer from the very
    // first frame, and what it must read is "I do not cover you". An unwritten
    // buffer would render as black ambient -- a hole, not a measurement.
    std::vector<RayFusion::ProbeTexel> zeroed(
        static_cast<size_t>(kMaxProbeSlots) * RayFusion::kProbeTexels);
    m_device->uploadBuffer(state->texels, zeroed.data(),
                           zeroed.size() * sizeof(RayFusion::ProbeTexel));
    ProbeGridParams params{};
    for (int axis = 0; axis < 3; ++axis) {
        params.counts[axis] = static_cast<int32_t>(state->counts[axis]);
        params.minimum[axis] = static_cast<int32_t>(state->minimum[axis]);
    }
    params.counts[3] = 0;
    params.minimum[3] = 0;
    params.spacing[0] = state->spacing;
    m_device->uploadBuffer(state->gridParams, &params, sizeof(params));

    state->ready = true;
    return true;
}

void VulkanBackendAdapter::setMaterialPreviewProbeSource(
    const float* equirectRgba, uint32_t width, uint32_t height, uint64_t signature) {
    if (!equirectRgba || width == 0u || height == 0u || signature == 0u) return;
    if (!ensureMaterialPreviewProbeResources()) return;
    auto state = m_materialPreviewProbes;
    if (state->sourceSignature == signature && !state->sourceIrradiance.empty()) return;
    const size_t floats = static_cast<size_t>(width) * height * 4u;
    state->sourceIrradiance.assign(equirectRgba, equirectRgba + floats);
    state->sourceWidth = width;
    state->sourceHeight = height;
    state->sourceSignature = signature;
}

// Izgarayi sahneye oturtur. IKI REJIM var ve hangisinin kostugu DURUM olarak
// yayinlanir -- "izgara neden bu kadar kaba" sorusunun cevabi bir tahmin
// olmamali.
//
//   scene        : trace edilen sahne butceye KULLANILABILIR yogunlukta
//                  siglyor. Izgara kutuya oturur, kamera takibi KAPANIR --
//                  iki otorite ayni degeri yazamaz.
//   camera_local : sahne cok buyuk (bu sahnede en yakin geometri 194 m).
//                  Butun butce kameranin etrafindaki kucuk bir hacme
//                  harcanir. Kaba bir izgarayla her yeri ortmek, ince bir
//                  izgarayla yakini ortmekten DAHA KOTUDUR: dolayli isik
//                  yerel bir olaydir ve uzaktaki probe zaten yanlis odada.
void VulkanBackendAdapter::applyMaterialPreviewProbeAutoFit(
    const RayFusionSceneASStatus& asStatus) {
    auto state = m_materialPreviewProbes;
    if (!state) return;

    float extent[3];
    bool sceneRegime = false;
    if (asStatus.world_bounds_valid) {
        bool finite = true;
        for (int axis = 0; axis < 3; ++axis) {
            extent[axis] = asStatus.world_max[axis] - asStatus.world_min[axis];
            if (!std::isfinite(extent[axis]) || extent[axis] < 0.0f) finite = false;
        }
        if (finite) {
            for (int axis = 0; axis < 3; ++axis) {
                // Duz bir zemin bir eksende SIFIR kalinliktadir; sifir bir
                // kutu, sifir bir aralik ve bolu sifir demektir.
                extent[axis] = (std::max)(extent[axis], 0.5f);
                extent[axis] *= (1.0f + 2.0f * kAutoFitPadFraction);
            }
            std::array<uint32_t, 3> probe{};
            const float trial = autoFitSpacingFor(extent, kMaxProbeSlots, probe);
            // Sahneyi ortebiliyoruz ama NE KADAR kabalikla? Yerel rejimin
            // verecegi yogunlugu asiyorsa sahneyi ortmek bir kazanc degil.
            std::array<uint32_t, 3> localProbe{};
            const float localSpacing =
                autoFitSpacingFor(kAutoFitLocalExtent, kMaxProbeSlots, localProbe);
            sceneRegime = trial <= localSpacing * 1.5f;
        }
    }

    float fitExtent[3];
    if (sceneRegime) {
        for (int axis = 0; axis < 3; ++axis) fitExtent[axis] = extent[axis];
    } else {
        for (int axis = 0; axis < 3; ++axis) fitExtent[axis] = kAutoFitLocalExtent[axis];
    }

    std::array<uint32_t, 3> counts{};
    const float spacing = autoFitSpacingFor(fitExtent, kMaxProbeSlots, counts);

    // Yerlesimin merkezi: sahne rejiminde kutunun ortasi, yerel rejimde kamera.
    float centre[3];
    if (sceneRegime) {
        for (int axis = 0; axis < 3; ++axis)
            centre[axis] = 0.5f * (asStatus.world_min[axis] + asStatus.world_max[axis]);
    } else {
        centre[0] = m_camera.origin.x;
        centre[1] = m_camera.origin.y;
        centre[2] = m_camera.origin.z;
        for (int axis = 0; axis < 3; ++axis)
            if (!std::isfinite(centre[axis]) || std::abs(centre[axis]) > 1e7f) return;
    }

    // ★★★ Yeniden oturtma GIRDI IMZASINA bagli. Her kare hesaplamak ucuz
    //   olurdu ama sonucu YAZMAK degil: sekil degisikligi `reconfigure` ile
    //   butun olcumleri dusurur, yani kamera her kimildandiginda alan bastan
    //   izlenir ve HICBIR ZAMAN yakinsamaz. Kamera 0,25 m'lik kovalara
    //   yuvarlaniyor, boylece kucuk hareket yeniden oturtmaz.
    uint64_t signature = 1469598103934665603ull;
    auto mix = [&signature](uint64_t v) {
        signature = (signature ^ v) * 1099511628211ull;
    };
    mix(sceneRegime ? 0x5CE9Eull : 0x10CA1ull);
    mix(static_cast<uint64_t>(static_cast<int64_t>(spacing * 100.0f)));
    for (int axis = 0; axis < 3; ++axis) {
        mix(counts[axis]);
        mix(static_cast<uint64_t>(static_cast<int64_t>(std::floor(centre[axis] * 4.0f))));
    }
    if (signature == state->autoFitSignature) return;
    state->autoFitSignature = signature;

    const RayFusion::Cell minimum = windowMinimumFor(centre, spacing, counts);
    const bool shapeChanged = counts != state->counts || spacing != state->spacing;
    state->counts = counts;
    state->spacing = spacing;
    state->minimum = minimum;
    if (shapeChanged) state->reconfigure = true;
    // Sahne rejiminde takip KAPANIR: izgara zaten butun sahneyi kapsiyor ve
    // kamerayla kaydirmak yalnizca olcumleri dusururdu. Yerel rejimde takip
    // ACIK olmali, yoksa pencere kameranin dogdugu yerde kalir.
    state->followCamera = !sceneRegime;
    state->autoFitMode = sceneRegime ? "scene" : "camera_local";
    if (sceneRegime) {
        state->autoFitReason = "traced scene fits the probe budget at usable density";
    } else if (!asStatus.world_bounds_valid) {
        state->autoFitReason = "scene AS published no world bounds; following the camera";
    } else {
        state->autoFitReason =
            "traced scene is too large to cover at usable density; spending the whole "
            "budget around the camera instead";
    }
    m_interactiveViewport.dirty = true;
}

void VulkanBackendAdapter::serviceMaterialPreviewProbeField() {
    auto state = m_materialPreviewProbes;
    if (state) state->pump = false;
    if (!state || !state->ready || !m_device) return;
    if (state->sourceSignature == 0u || state->sourceIrradiance.empty()) return;

    // ★★★ The traced producer depends on GEOMETRY as well as on the sky, so the
    //   revision must carry both. With only the sky signature, moving a wall
    //   would leave every probe holding the occlusion of a wall that is no
    //   longer there -- valid, unchanged, and wrong, with no error anywhere.
    RayFusionSceneASStatus asStatus{};
    const bool asReady = getRayFusionSceneASStatus(asStatus) && asStatus.ready;
    const bool useTraced = state->preferTraced && asReady;
    const uint64_t bounceSignature = prepareRayFusionBounce();

    // ★★★★ Otomatik yerlesim, alan YAPILANDIRILMADAN once calisir: bir sekil
    //   degisikligi zaten `reconfigure` ister, ve ikisini ayni gecise koymak
    //   alanin ayni karede iki kez gecersiz kilinmasini onler.
    if (state->autoFit) applyMaterialPreviewProbeAutoFit(asStatus);

    uint64_t producerSignature = state->sourceSignature;
    if (useTraced) {
        producerSignature ^= 0x9E3779B97F4A7C15ull;
        producerSignature = producerSignature * 1099511628211ull ^ asStatus.geometry_signature;
        producerSignature = producerSignature * 1099511628211ull ^ asStatus.instance_signature;
        producerSignature = producerSignature * 1099511628211ull ^ bounceSignature;
    }

    // The producer signature IS the scene epoch here. When the sun moves the
    // bake is regenerated, the signature changes, and the whole field is
    // invalidated -- old light is never carried into a new sky.
    RayFusion::Revision revision;
    revision.deviceEpoch = 1;
    revision.sceneEpoch = producerSignature;
    revision.lighting = producerSignature;
    revision.geometry = useTraced ? asStatus.geometry_signature : 0u;

    std::string error;
    // A shape change (counts or spacing) rebuilds the slot array and drops every
    // measurement, because the cells themselves are different volumes of world.
    // Carrying the old packets across would keep a value measured for a
    // different place -- valid, unchanged and wrong, with no error anywhere.
    if (!state->configured || state->reconfigure) {
        RayFusion::Grid grid;
        grid.counts = state->counts;
        grid.minimum = state->minimum;
        grid.spacing = state->spacing;
        grid.targetUpdates = 1; // one producer, one valid result per slot
        if (!state->field.configure(grid, revision, error)) {
            SCENE_LOG_WARN(std::string("[RayFusion] probe field configure failed: ") + error);
            state->reconfigure = false; // do not retry a rejected grid every frame
            // Roll the request back to what is actually applied. Leaving the two
            // apart would make status report one window while the scroll check
            // below fought to reach another one, every frame, forever.
            if (state->configured) {
                const auto& applied = state->field.grid();
                state->counts = applied.counts;
                state->spacing = applied.spacing;
                state->minimum = applied.minimum;
            }
            return;
        }
        state->configured = true;
        state->reconfigure = false;
        state->uploaded = false;
        state->appliedRevision = producerSignature;
    } else if (state->appliedRevision != producerSignature) {
        if (!state->field.invalidate(revision, error)) {
            SCENE_LOG_WARN(std::string("[RayFusion] probe field invalidate failed: ") + error);
            return;
        }
        state->appliedRevision = producerSignature;
        state->uploaded = false;
    }

    if (state->followCamera) {
        const auto& grid = state->field.grid();
        const float position[3] = {m_camera.origin.x, m_camera.origin.y, m_camera.origin.z};
        // Checked BEFORE any arithmetic: floor(inf / spacing) cast to an integer
        // is undefined, and the camera is one bad frame away from carrying a NaN.
        for (int axis = 0; axis < 3; ++axis)
            if (!std::isfinite(position[axis]) || std::abs(position[axis]) > 1e7f) return;
        RayFusion::Cell minimum = grid.minimum;
        const RayFusion::Cell centred = windowMinimumFor(position, grid.spacing, grid.counts);
        for (int axis = 0; axis < 3; ++axis) {
            const int64_t centre = static_cast<int64_t>(std::floor(position[axis] / grid.spacing));
            // A full-cell dead band avoids boundary jitter. Camera rotation
            // does not change this anchor; only newly exposed cells reset.
            if (std::abs(centred[axis] - minimum[axis]) > 1 || centre < minimum[axis] ||
                centre >= minimum[axis] + static_cast<int64_t>(grid.counts[axis]))
                minimum[axis] = centred[axis];
        }
        if (minimum != grid.minimum) {
            if (!state->field.scroll(minimum, error)) return;
            state->uploaded = false;
        }
        // Following OWNS the placement while it is on, so the request has to
        // follow it. Otherwise turning following off would snap the window back
        // to wherever the last explicit placement had been -- a jump nobody
        // asked for, at the moment the user asked for the window to FREEZE.
        state->minimum = state->field.grid().minimum;
    } else if (state->minimum != state->field.grid().minimum) {
        // Explicit placement. scroll() keeps the world cells that stayed inside
        // the window; only newly exposed cells are rescheduled.
        if (!state->field.scroll(state->minimum, error)) {
            SCENE_LOG_WARN(std::string("[RayFusion] probe field scroll failed: ") + error);
            state->minimum = state->field.grid().minimum;
        } else {
            state->uploaded = false;
        }
    }
    auto budget = RayFusion::budgetForQuality(qualityPresetName());
    const auto bounce = rayFusionBounceStatus();
    // A hit can issue one environment ray and one light shadow ray in addition
    // to the primary. Keep the total ray ceiling, not merely the primary count.
    if (useTraced) budget.raysPerProbe = bounce.requested && bounce.ready ? 192u : 64u;
    auto tickets = state->field.schedule(budget);
    if (tickets.empty()) return;

    uint32_t accepted = 0;
    bool tracedThisBatch = false;
    state->producerReason.clear();

    // ── Producer A: traced (step 1b-alpha) ─────────────────────────────────
    if (useTraced) {
        std::vector<float> origins(tickets.size() * 4u, 0.0f);
        for (size_t i = 0; i < tickets.size(); ++i) {
            // The consumer derives the cell as floor(worldPos / spacing), so a
            // cell spans [c*spacing, (c+1)*spacing) and its probe sits at the
            // centre. Any other origin would measure a place no pixel reads.
            const auto& cell = tickets[i].cell;
            const float spacing = state->field.grid().spacing;
            origins[i * 4u + 0u] = (static_cast<float>(cell[0]) + 0.5f) * spacing;
            origins[i * 4u + 1u] = (static_cast<float>(cell[1]) + 0.5f) * spacing;
            origins[i * 4u + 2u] = (static_cast<float>(cell[2]) + 0.5f) * spacing;
        }

        std::vector<RayFusion::ProbeTexel> traced;
        if (traceRayFusionProbes(origins.data(), static_cast<uint32_t>(tickets.size()),
                                 traced) &&
            traced.size() >= tickets.size() * RayFusion::kProbeTexels) {
            tracedThisBatch = true;
            double hitSum = 0.0;
            double meanDistanceSum = 0.0;
            for (size_t i = 0; i < tickets.size(); ++i) {
                RayFusion::ProbePacket packet{};
                std::memcpy(packet.data(), &traced[i * RayFusion::kProbeTexels],
                            sizeof(RayFusion::ProbePacket));
                // The shader parks its two per-probe statistics in the reserved
                // distance lanes; read them, then put the lanes back to zero so
                // what is stored matches the documented ABI exactly.
                const float backFraction = packet[0].distance[2];
                const float hitFraction = packet[0].distance[3];
                hitSum += hitFraction;
                // Yonler uzerinden ortalama carpma mesafesi: probe'un icinde
                // durdugu bosugun OLCEGI.
                double distSum = 0.0;
                for (const auto& t : packet) distSum += t.distance[0];
                const float meanHitDistance =
                    static_cast<float>(distSum / static_cast<double>(packet.size()));
                const float insideRadius =
                    state->field.grid().spacing * kInsideGeometryDistanceFraction;
                const bool inside = backFraction > kInsideGeometryBackfaceFraction &&
                                    meanHitDistance < insideRadius;
                meanDistanceSum += meanHitDistance;
                if (backFraction > kInsideGeometryBackfaceFraction && !inside)
                    ++state->backfaceEnclosed;
                for (auto& texel : packet) {
                    texel.distance[2] = 0.0f;
                    texel.distance[3] = 0.0f;
                    // Alpha 0 = "no usable measurement here". The slot still
                    // becomes VALID so it is not re-traced every frame, and
                    // rfSampleProbeField returns false for it, so the pixel
                    // keeps the global read instead of going black.
                    if (inside) {
                        texel.irradiance = {0.0f, 0.0f, 0.0f, 0.0f};
                    }
                }
                if (inside) ++state->rejectedInside;
                if (state->field.publish(tickets[i], packet, 0.0f, error)) ++accepted;
            }
            state->lastHitFraction = tickets.empty()
                ? 0.0f : static_cast<float>(hitSum / static_cast<double>(tickets.size()));
            state->lastMeanHitDistance = tickets.empty()
                ? 0.0f : static_cast<float>(meanDistanceSum / static_cast<double>(tickets.size()));
            ++state->tracedPublishes;
            state->activeProducer = "traced";
            state->activeBounce = bounce.requested && bounce.ready;
        } else {
            bool traceSupported = false;
            uint64_t dispatches = 0, probesTraced = 0;
            double lastMs = 0.0;
            std::string reason;
            getRayFusionProbeTraceStatus(traceSupported, lastMs, dispatches,
                                         probesTraced, reason);
            state->producerReason = reason.empty() ? "probe trace dispatch failed" : reason;
        }
    } else if (state->preferTraced) {
        state->producerReason = "scene acceleration structure is not ready";
    }

    // ── Producer B: sky bake (step 1a) ─────────────────────────────────────
    if (!tracedThisBatch) {
        // One packet for every probe: sky irradiance per direction, identical
        // in every cell because no visibility is measured on this path.
        RayFusion::ProbePacket packet{};
        for (uint32_t y = 0; y < RayFusion::kProbeSide; ++y)
            for (uint32_t x = 0; x < RayFusion::kProbeSide; ++x) {
                const float ex = ((static_cast<float>(x) + 0.5f) /
                                  static_cast<float>(RayFusion::kProbeSide)) * 2.0f - 1.0f;
                const float ey = ((static_cast<float>(y) + 0.5f) /
                                  static_cast<float>(RayFusion::kProbeSide)) * 2.0f - 1.0f;
                float direction[3];
                octDecode(ex, ey, direction);
                float rgb[3];
                sampleEquirect(state->sourceIrradiance, state->sourceWidth,
                               state->sourceHeight, direction, rgb);
                auto& texel = packet[y * RayFusion::kProbeSide + x];
                texel.irradiance = {std::max(0.0f, rgb[0]), std::max(0.0f, rgb[1]),
                                    std::max(0.0f, rgb[2]), 1.0f};
                texel.distance = {kUnoccludedDistance,
                                  kUnoccludedDistance * kUnoccludedDistance, 0.0f, 0.0f};
            }
        for (const auto& ticket : tickets) {
            // historyWeight 0: this producer is static, so there is no noise to
            // filter. A smoothing factor here would only add lag.
            if (state->field.publish(ticket, packet, 0.0f, error)) ++accepted;
        }
        state->activeProducer = "sky_bake";
        state->activeBounce = false;
        state->lastHitFraction = 0.0f;
    }
    if (accepted == 0u) return;

    // Write the GPU buffer only when new slots were actually published. Writing
    // every frame would mutate unchanged data underneath an in-flight frame.
    // Sized by the ACTIVE window, not by the allocation ceiling: a 32-slot grid
    // must not pay a 2 MiB copy because a 1024-slot one is allowed.
    const auto& grid = state->field.grid();
    const uint32_t slotCount = slotsFor(grid.counts);
    std::vector<RayFusion::ProbeTexel> upload(
        static_cast<size_t>(slotCount) * RayFusion::kProbeTexels);
    uint32_t validSlots = 0;
    for (uint32_t z = 0; z < grid.counts[2]; ++z)
        for (uint32_t y = 0; y < grid.counts[1]; ++y)
            for (uint32_t x = 0; x < grid.counts[0]; ++x) {
                const auto& minimum = grid.minimum;
                const RayFusion::Cell cell{minimum[0] + static_cast<int64_t>(x),
                                           minimum[1] + static_cast<int64_t>(y),
                                           minimum[2] + static_cast<int64_t>(z)};
                const auto* stored = state->field.lookup(cell);
                if (!stored) continue;
                const uint32_t slot = state->field.slotIndex(cell);
                if (slot >= slotCount) continue;
                std::memcpy(&upload[static_cast<size_t>(slot) * RayFusion::kProbeTexels],
                            stored->data(), sizeof(RayFusion::ProbePacket));
                ++validSlots;
            }

    drainInteractiveViewportInFlight();
    m_device->uploadBuffer(state->texels, upload.data(),
                           upload.size() * sizeof(RayFusion::ProbeTexel));

    ProbeGridParams params{};
    for (int axis = 0; axis < 3; ++axis) {
        params.counts[axis] = static_cast<int32_t>(grid.counts[axis]);
        params.minimum[axis] = static_cast<int32_t>(grid.minimum[axis]);
    }
    params.counts[3] = static_cast<int32_t>(slotCount);
    params.minimum[3] = validSlots > 0u ? 1 : 0;
    params.spacing[0] = grid.spacing;
    // Only actual traced publications carry measured sky visibility. A request
    // that fell back to sky bake must never enable this consumer.
    params.spacing[1] = state->activeProducer == "traced"
        ? RayFusion::kProbeTraceDistance : 0.0f;
    m_device->uploadBuffer(state->gridParams, &params, sizeof(params));
    state->uploaded = validSlots > 0u;
    state->pump = state->field.stats().pending > 0u;
}

bool VulkanBackendAdapter::bindMaterialPreviewProbeDescriptors(VkDescriptorSet set) {
    // Create the buffers here, not only when the first bake lands. The fragment
    // shader declares bindings 21/22 from the first frame, and a descriptor the
    // shader statically uses but nobody ever wrote is undefined behaviour --
    // this codebase has already paid for that once, at bindings 5/6.
    if (!ensureMaterialPreviewProbeResources()) return false;
    auto state = m_materialPreviewProbes;
    if (!m_device || set == VK_NULL_HANDLE || !state || !state->ready) return false;
    if (state->boundSet == set) return state->uploaded;

    drainInteractiveViewportInFlight();
    VkDescriptorBufferInfo buffers[2]{};
    buffers[0].buffer = state->texels.buffer;
    buffers[0].range = VK_WHOLE_SIZE;
    buffers[1].buffer = state->gridParams.buffer;
    buffers[1].range = VK_WHOLE_SIZE;
    VkWriteDescriptorSet writes[2]{};
    for (uint32_t i = 0; i < 2; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = 21u + i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buffers[i];
    }
    vkUpdateDescriptorSets(m_device->getDevice(), 2, writes, 0, nullptr);
    state->boundSet = set;
    return state->uploaded;
}

void VulkanBackendAdapter::setRayFusionProbeProducer(bool traced) {
    if (!ensureMaterialPreviewProbeResources()) return;
    auto state = m_materialPreviewProbes;
    if (!state || state->preferTraced == traced) return;
    state->preferTraced = traced;
    // Switching producers must NOT blend two worlds. The revision is rebuilt
    // from the new producer's inputs on the next service call, which invalidates
    // every slot; without that, half the field would still hold the other
    // producer's answer and the A/B comparison would measure a mixture.
    state->appliedRevision = 0;
    state->rejectedInside = 0;
    // Bu da sifirlanmali: aksi halde yeni ureticinin ilk okumasi ESKI
    // ureticinin sayisini gosterir ve A/B karsilastirmasi karisimi olcer.
    state->backfaceEnclosed = 0;
    state->lastMeanHitDistance = 0.0f;
    state->lastHitFraction = 0.0f;
    state->producerReason.clear();
    // Ayni gerekce: gecersiz kilmak yeni bir kare cizilmeden hicbir sey yapmaz.
    // Ayrintili aciklama RayFusionBounce.cpp / setRayFusionProbeBounce.
    m_interactiveViewport.dirty = true;
}

bool VulkanBackendAdapter::getRayFusionProbeStatus(RayFusionProbeStatus& out) const {
    out = {};
    out.bounce = rayFusionBounceStatus();
    const auto state = m_materialPreviewProbes;
    if (!state) return false;
    out.supported = state->ready;
    out.overlay_requested = state->overlay;
    out.overlay_ready = state->overlayReady;
    out.overlay_markers = state->overlayMarkers;
    out.overlay_reason = state->overlayReason;
    out.follow_camera = state->followCamera;
    out.auto_fit = state->autoFit;
    out.auto_fit_mode = state->autoFitMode;
    out.auto_fit_reason = state->autoFitReason;
    out.backface_enclosed = state->backfaceEnclosed;
    out.mean_hit_distance = state->lastMeanHitDistance;
    out.configured = state->configured;
    out.uploaded = state->uploaded;
    out.bound = state->boundSet != VK_NULL_HANDLE;
    out.producer = state->sourceSignature != 0u ? state->activeProducer : "none";
    out.bounce.active = state->activeBounce && state->uploaded && out.producer == "traced";
    out.producer_traced_requested = state->preferTraced;
    out.producer_reason = state->producerReason;
    out.producer_signature = state->appliedRevision;
    out.hit_fraction = state->lastHitFraction;
    out.rejected_inside = state->rejectedInside;
    out.traced_publishes = state->tracedPublishes;
    {
        bool traceSupported = false;
        uint64_t dispatches = 0, probesTraced = 0;
        double lastMs = 0.0;
        std::string reason;
        if (getRayFusionProbeTraceStatus(traceSupported, lastMs, dispatches,
                                         probesTraced, reason)) {
            out.trace_ms = lastMs;
            if (out.producer_reason.empty()) out.producer_reason = reason;
        }
    }
    out.budget_preset = qualityPresetName();
    out.max_slots = kMaxProbeSlots;
    if (state->configured) {
        const auto stats = state->field.stats();
        out.total = stats.total;
        out.valid = stats.valid;
        out.pending = stats.pending;
        out.in_flight = stats.inFlight;
        out.accepted = stats.accepted;
        out.rejected = stats.rejected;
        const auto& grid = state->field.grid();
        for (uint32_t i = 0; i < 3; ++i) {
            out.counts[i] = grid.counts[i];
            out.minimum[i] = static_cast<int32_t>(grid.minimum[i]);
        }
        out.spacing = grid.spacing;
    }
    return true;
}

bool VulkanBackendAdapter::setRayFusionProbeOverlay(bool enabled) {
    if (!ensureMaterialPreviewProbeResources()) return false;
    auto& state = *m_materialPreviewProbes;
    state.overlay = enabled;
    state.overlayReady = false;
    state.overlayMarkers = 0;
    state.overlayReason = enabled ? "Waiting for a raster viewport frame" : "";
    m_interactiveViewport.dirty = true;
    return true;
}

bool VulkanBackendAdapter::setRayFusionProbeFollowCamera(bool enabled) {
    if (!ensureMaterialPreviewProbeResources()) return false;
    // Takip de bir YERLESIM otoritesi; otomatik oturtma ile ikisi ayni degeri
    // yazar. Elle takip istegi otomatigi kapatir, aksi halde istek bir kare
    // sonra geri alinirdi.
    if (m_materialPreviewProbes->autoFit) {
        m_materialPreviewProbes->autoFit = false;
        m_materialPreviewProbes->autoFitMode = "off";
        m_materialPreviewProbes->autoFitReason =
            "follow-camera was set explicitly and owns placement";
    }
    m_materialPreviewProbes->followCamera = enabled;
    m_interactiveViewport.dirty = true;
    return true;
}

bool VulkanBackendAdapter::setRayFusionProbeGrid(const RayFusion::GridRequest& request,
                                                 std::string& error) {
    error.clear();
    if (!ensureMaterialPreviewProbeResources()) {
        error = "probe field resources are not available on this backend";
        return false;
    }
    auto& state = *m_materialPreviewProbes;

    // Resolve against the CURRENT window first, then validate the RESULT. A
    // caller that sends only counts must not have to know the spacing, and the
    // pair has to be checked together anyway -- 16x16x16 is legal, 3.0 spacing
    // is legal, and the product of the two is not.
    auto counts = state.counts;
    float spacing = state.spacing;
    RayFusion::Cell minimum = state.minimum;

    if (request.hasCounts) {
        for (int axis = 0; axis < 3; ++axis) {
            if (request.counts[axis] < 1u || request.counts[axis] > 128u) {
                error = "each grid count must be 1..128";
                return false;
            }
            counts[axis] = request.counts[axis];
        }
    }
    if (request.hasSpacing) {
        if (!std::isfinite(request.spacing) || request.spacing <= 0.0f ||
            request.spacing > 1.0e4f) {
            error = "spacing must be finite, positive and at most 10000 world units";
            return false;
        }
        spacing = request.spacing;
    }
    const uint64_t slots = static_cast<uint64_t>(counts[0]) * counts[1] * counts[2];
    if (slots > kMaxProbeSlots) {
        error = "grid needs " + std::to_string(slots) + " probe slots; the GPU buffer holds " +
                std::to_string(kMaxProbeSlots);
        return false;
    }
    if (request.hasMinimum && request.hasCenter) {
        error = "send either 'minimum' (cells) or 'center' (world units), not both";
        return false;
    }
    if (request.hasMinimum) {
        for (int axis = 0; axis < 3; ++axis) minimum[axis] = request.minimum[axis];
    } else if (request.hasCenter) {
        for (int axis = 0; axis < 3; ++axis) {
            if (!std::isfinite(request.center[axis]) ||
                std::abs(request.center[axis]) > 1.0e7f) {
                error = "center must be finite and within 1e7 world units";
                return false;
            }
        }
        // Resolved with the FINAL spacing and counts, which is exactly why the
        // caller cannot do this itself in a call that also changes spacing.
        minimum = windowMinimumFor(request.center.data(), spacing, counts);
    }
    for (int axis = 0; axis < 3; ++axis) {
        const double extent = (std::abs(static_cast<double>(minimum[axis])) +
                               static_cast<double>(counts[axis])) * spacing;
        if (extent > 1.0e9) {
            error = "grid reaches beyond the supported world range";
            return false;
        }
    }

    // An explicit placement and camera following are two authorities over the
    // same value. Rather than let following silently overwrite the placement on
    // the next frame -- the request would appear to have been accepted and then
    // do nothing -- following is turned OFF, which is visible in probe_field.
    // Unconditional, not "only if the window actually moved": a placement that
    // happens to resolve to the cell the camera is already over would otherwise
    // leave following on, and the window would drift away a second later.
    if (request.hasMinimum || request.hasCenter) state.followCamera = false;
    // ★★★★ Ayni gerekce otomatik yerlesim icin de gecerli ve DAHA guclu:
    //   otomatik oturtma her serviste sekli de yerlesimi de yazar, yani elle
    //   gonderilen bir izgara kabul edilir, bir kare yasar ve sessizce geri
    //   alinirdi. "Istek reddedildi" gibi degil, "istek unutuldu" gibi
    //   gorunurdu -- hicbir hata mesaji olmadan.
    if (request.hasAutoFit) {
        state.autoFit = request.autoFit;
        if (!request.autoFit) {
            state.autoFitMode = "off";
            state.autoFitReason = "turned off by request";
        }
        state.autoFitSignature = 0;   // yeniden acilirsa hemen otursun
    } else if (request.hasCounts || request.hasSpacing ||
               request.hasMinimum || request.hasCenter) {
        state.autoFit = false;
        state.autoFitMode = "off";
        state.autoFitReason = "an explicit grid request took over placement";
    }

    const bool shapeChanged = counts != state.counts || spacing != state.spacing;
    const bool placementChanged = minimum != state.minimum;
    if (!shapeChanged && !placementChanged) {
        m_interactiveViewport.dirty = true;   // following may have just been dropped
        return true;                          // idempotent, not an error
    }

    state.counts = counts;
    state.spacing = spacing;
    state.minimum = minimum;
    if (shapeChanged) state.reconfigure = true;
    // The window is only applied while the field is being serviced, and that
    // happens when the viewport produces a frame. Without this the call would
    // be accepted and the next status read would report the OLD grid, which
    // reads exactly like a rejected request.
    m_interactiveViewport.dirty = true;
    return true;
}

bool VulkanBackendAdapter::rayFusionProbeUpdatesPending() const {
    return m_materialPreviewProbes && m_materialPreviewProbes->pump;
}

bool VulkanBackendAdapter::rayFusionProbeOverlayRequested() const {
    return m_materialPreviewProbes && m_materialPreviewProbes->overlay;
}

std::vector<RayFusion::ProbeMarker> VulkanBackendAdapter::rayFusionProbeMarkers() const {
    std::vector<RayFusion::ProbeMarker> markers;
    auto state = m_materialPreviewProbes;
    if (!state || !state->overlay || !state->configured) return markers;
    const auto& grid = state->field.grid();
    markers.reserve(slotsFor(grid.counts));
    for (uint32_t z = 0; z < grid.counts[2]; ++z)
    for (uint32_t y = 0; y < grid.counts[1]; ++y)
    for (uint32_t x = 0; x < grid.counts[0]; ++x) {
        RayFusion::Cell cell{grid.minimum[0] + x, grid.minimum[1] + y, grid.minimum[2] + z};
        RayFusion::ProbeMarker marker;
        for (int axis = 0; axis < 3; ++axis)
            marker.position[axis] = (static_cast<float>(cell[axis]) + 0.5f) * grid.spacing;
        const auto* packet = state->field.lookup(cell);
        marker.state = !packet ? 0u : ((*packet)[0].irradiance[3] >= 0.5f ? 1u : 2u);
        markers.push_back(marker);
    }
    return markers;
}

void VulkanBackendAdapter::setRayFusionProbeOverlayResult(bool ready, uint32_t count,
                                                         const std::string& reason) {
    if (!m_materialPreviewProbes) return;
    m_materialPreviewProbes->overlayReady = ready;
    m_materialPreviewProbes->overlayMarkers = count;
    m_materialPreviewProbes->overlayReason = reason;
}

void VulkanBackendAdapter::destroyMaterialPreviewProbeResources() {
    // Torn down FIRST and unconditionally: the trace pass holds a descriptor
    // set pointing at the TLAS and at the probe result buffer. Leaving it alive
    // past the field it serves would keep a set bound to freed memory, and a
    // stale acceleration-structure descriptor does not fault here -- it traces
    // something that no longer exists.
    destroyRayFusionProbeTraceResources();
    destroyRayFusionBounce();
    if (!m_materialPreviewProbes || !m_device) {
        m_materialPreviewProbes.reset();
        return;
    }
    auto& state = *m_materialPreviewProbes;
    if (state.texels.buffer) m_device->destroyBuffer(state.texels);
    if (state.gridParams.buffer) m_device->destroyBuffer(state.gridParams);
    m_materialPreviewProbes.reset();
}

} // namespace Backend
