/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Camera.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#ifndef CAMERA_H
#define CAMERA_H

#include <vector>
#include <cmath>   // exposureFactor(): std::pow
#include "Vec3.h"
#include "Matrix4x4.h"
#include "Ray.h"
#include "AABB.h"
#include "ThreadLocalRNG.h"
#include "CameraPresets.h"   // exposureFactor(): ISO/shutter/f-stop tablolari

// ═══════════════════════════════════════════════════════════════════════════════
// CAMERA MODE - Controls feature availability and physical simulation level
// ═══════════════════════════════════════════════════════════════════════════════
enum class CameraMode {
    Auto,       // Amatör - Otomatik ayarlar, kısıtlı kontrol, kusurlar kapalı
    Pro,        // Profesyonel - Manuel kontrol, temiz görüntü, opsiyonel kusurlar  
    Cinema      // Sinematik - Tam fiziksel simülasyon, tüm lens/sensör kusurları
};
class Camera {
private:
    struct Plane {
        Vec3 normal;
        float distance;

        Plane() : normal(Vec3()), distance(0.0f) {}
        Plane(const Vec3& n, const Vec3& point) : normal(n.normalize()) {
            distance = -Vec3::dot(normal, point);
        }

        float distanceToPoint(const Vec3& point) const {
            return Vec3::dot(normal, point) + distance;
        }
    };

public:
    Vec3 initialLookDirection;
    std::string nodeName;
    bool visible = true;
    int blade_count = 6;
    float aperture = 0.0f;

    // ★★★★★ ALAN DERINLIGININ ACIK OLMASI, `aperture`in KENDISI DEGILDIR.
    //   2026-09-06'ya kadar tek kapali-anahtar `aperture == 0` idi. Bu bir
    //   DEGER degil bir SENTINEL'di, ve f-sayisi kadranini buraya baglar
    //   baglamaz oldu: hicbir f-sayisi sifir aciklik uretmez, yani geri
    //   donusun TEK yolu yok oldu. Kullanicinin belirtisi tam olarak buydu --
    //   "f-stop'a bir kez dokundum, DoF bir daha kapanmiyor".
    //   ★★ Yeni sozlesme:
    //     `aperture`        = her zaman FIZIKSEL aciklik (f-sayisinin ikizi),
    //     `depth_of_field`  = lens diski ORNEKLENSIN mi.
    //   Kapatmak degeri yok etmez; geri acmak ayni bulanikligi aninda verir.
    //   ★★★ Ve kapiyi tuketiciler tek tek kurmaz: hepsi
    //   `effectiveLensRadius()` / `effectiveAperture()` uzerinden gecer. Ayni
    //   "AND"i dort backend'e kopyalamak bu deponun adi konmus hata sinifi.
    bool depth_of_field = false;

    float focus_dist = 10.0f;
    Vec3 origin;
    Vec3 u, v, w;
    Vec3 lookfrom;
    Vec3 lookat;
    Vec3 vup;
    float aspect = 1.7777f;
    float near_dist = 0.01f;
    float far_dist = 20000.0f;
    float fov = 45.0f;
    float aspect_ratio = 1.7777f;
    float vfov = 45.0f;

    // ═══════════════════════════════════════════════════════════════════════════
    // ORTHOGRAPHIC / STANDARD VIEWS (viewport alignment)
    // ═══════════════════════════════════════════════════════════════════════════
    // When orthographic is true, primary rays are parallel (no perspective foreshortening),
    // which removes the depth illusion that makes aligning objects unreliable.
    enum class StandardView { Perspective, Top, Bottom, Front, Back, Left, Right };
    bool orthographic = false;       // true = parallel projection
    float ortho_height = 10.0f;      // full vertical extent (world units) visible at the lookat plane
    StandardView standard_view = StandardView::Perspective;

    // ═══════════════════════════════════════════════════════════════════════════
    // ORBIT PIVOT (viewport navigation anchor)
    // ═══════════════════════════════════════════════════════════════════════════
    // ★★★★★ THE PIVOT IS NOT THE FOCAL PLANE. Until 2026-09-23 there was no
    //   pivot field: `lookat` played the part, and three unrelated writers
    //   fought over it every frame --
    //     frame_selected()  wrote the selection centre,
    //     the middle-click raycast wrote whatever surface sat under the cursor,
    //     setLookDirection() wrote `lookfrom + dir * focus_dist`.
    //   The third one is the expensive one, and it explains a symptom that read
    //   as unrelated: pulling the focus ring to 0.5 m made panning and dollying
    //   crawl. `focus_dist` is a LENS property (depth of field, autofocus); the
    //   moment it also set the navigation radius, focusing near re-anchored the
    //   whole viewport 0.5 m in front of the camera. Nothing errored -- the
    //   camera simply stopped responding, which nobody reports as a bug.
    //
    //   So the two live in separate fields now, and nothing below ever writes
    //   `focus_dist`. Autofocus stays the only writer of the focal plane.
    //
    // ★★ Free vs Selection is a question of WHO OWNS THE ANCHOR, not of style:
    //   - Free      : the anchor follows navigation (cursor raycast, fly-look).
    //   - Selection : the anchor is the selection's bounds centre and survives
    //                 orbit, dolly and pan. Re-armed by Frame Selected.
    enum class PivotMode { Free, Selection };
    PivotMode pivot_mode = PivotMode::Free;
    Vec3 orbit_pivot;                 // world-space anchor
    bool pivot_valid = false;         // false -> effectivePivot() falls back to lookat

    // The anchor the navigation actually uses. Falls back to `lookat` so every
    // caller has one question to ask, and an unarmed pivot reproduces the old
    // behaviour instead of snapping to the origin.
    Vec3 effectivePivot() const { return pivot_valid ? orbit_pivot : lookat; }

    // Distance orbit/dolly work at and screen-correct pan is measured from.
    // ★ Never `focus_dist` -- see the block above.
    float navDistance() const;

    // Arm/disarm the anchor. Neither moves the camera: arming a pivot must not
    // jump the view, or Frame Selected could not be composed from these.
    void setOrbitPivot(const Vec3& p) { orbit_pivot = p; pivot_valid = true; }
    void clearOrbitPivot() { pivot_valid = false; }

    // Orbit the camera around the effective pivot (world-Y yaw, camera-right
    // pitch, degrees). `lookat` is re-aimed at the pivot: an orbit that does not
    // look at what it turns around is a fly-look wearing an orbit's name.
    void orbitAroundPivot(float dyaw_deg, float dpitch_deg);

    // Dolly toward (negative) / away from (positive) the pivot. Exponential so
    // the perceptual response is the same from centimetre detail to kilometre
    // terrain, and the camera can never cross the anchor.
    void dollyToPivot(float exponent);

    // Translate camera, target AND pivot by the same offset, so panning slides
    // the view without breaking the lock or changing the navigation radius.
    void panWorld(const Vec3& offset);

    // Snap the camera to a standard axis-aligned view around the current lookat (pivot).
    // Preserves the current distance and frames the same world span (continuous switch).
    // setOrtho=true also flips to parallel projection (the usual DCC behaviour for these views).
    void setStandardView(StandardView v, bool setOrtho = true);

    // Same, but explicitly orbit around `pivot` at `distance` (so the snap can re-centre on the
    // selection / world origin instead of whatever stale point lookat happened to hold).
    void setStandardView(StandardView v, const Vec3& pivot, float distance, bool setOrtho = true);

    // Camera(Vec3 lookfrom, Vec3 lookat, Vec3 vup, double vfov, double aspect, double aperture, double focus_dist);

    Camera(Vec3 lookfrom, Vec3 lookat, Vec3 vup, float vfov, float aspect, float aperture, float focus_dist, int blade_count);
    Camera();
    Ray get_ray(float s, float t) const;
    // Deterministic viewport/picking ray. Raster projection is pinhole/ortho:
    // it does not apply render-lens distortion or stochastic aperture offsets.
    Ray get_viewport_ray(float s, float t) const;

    int random_int(int min, int max) const;

    void update_camera_vectors();

    void moveToTargetLocked(const Vec3& new_position);

    void setLookDirection(const Vec3& direction_normalized);

    Vec3 random_in_unit_polygon(int sides) const;

    float calculate_bokeh_intensity(const Vec3& point) const;

    Vec3 create_bokeh_shape(const Vec3& color, float intensity) const;
    void reset();
    void save_initial_state();
    bool isPointInFrustum(const Vec3& point, float size) const;
    Matrix4x4 getRotationMatrix() const;
    bool isAABBInFrustum(const AABB& aabb) const;
    std::vector<AABB> performFrustumCulling(const std::vector<AABB>& objects) const;
    Vec3 lower_left_corner;
    Vec3 horizontal;
    Vec3 vertical;
    float lens_radius = 0.0f;   // FIZIKSEL yaricap (= aperture * 0.5). Kapi DEGIL.

    // ★★★ TEK TANIM: "lens gercekten orneklenecek mi" sorusunun cevabi.
    //   Ortografik kamerada lens yoktur; onu cagiran taraf ayrica eler cunku
    //   ortografik yol zaten farkli bir isin uretimidir.
    float effectiveAperture() const { return depth_of_field ? aperture : 0.0f; }
    float effectiveLensRadius() const { return depth_of_field ? aperture * 0.5f : 0.0f; }

    // ── f-sayisi <-> aciklik: TEK EGRI, iki yonu de birbirinin TERSI ────────
    // ★★★★★ 2026-09-06'ya kadar UC ayri donusum vardi ve UCU DE farkli sayi
    //   veriyordu:
    //     preset combo'su : FSTOP_PRESETS[i].aperture_value  (f/2.8 -> 1.20)
    //     f-stop slider'i : (focal_mm / f) * 0.01            (f/2.8 -> 0.18)
    //     panel gosterimi : focal_mm / aperture              (0.18  -> f/280!)
    //   Yani combo'dan f/2.8 secip slider'i oynatmak bulanikligi 7 KAT
    //   degistiriyordu ve panelin gosterdigi f-sayisi ust sinira yapisiyordu.
    //   Hicbiri hata vermiyordu -- yalnizca "kadran tuhaf davraniyor".
    //
    //   ★★★ Cozum: OTORITE PRESET TABLOSUDUR. Tablo elle ayarlanmis sanatsal
    //   bir olcektir (1/f DEGIL), o yuzden ara degerler tabloyu log-log
    //   INTERPOLE eder. Boylece preset degerlerinde gorunum AYNEN korunur --
    //   tabloyu bir formulle degistirmek her mevcut sahnenin bulanikligini
    //   sessizce degistirirdi.
    static float apertureForFNumber(float f_number) {
        const int n = (int)CameraPresets::FSTOP_PRESET_COUNT;
        if (n < 3) return 0.0f;                    // 0 = "Custom", egri 1..n-1
        const float f_lo = CameraPresets::FSTOP_PRESETS[1].f_number;
        const float f_hi = CameraPresets::FSTOP_PRESETS[n - 1].f_number;
        if (f_number <= f_lo) return CameraPresets::FSTOP_PRESETS[1].aperture_value;
        if (f_number >= f_hi) return CameraPresets::FSTOP_PRESETS[n - 1].aperture_value;
        for (int i = 1; i < n - 1; ++i) {
            const float a = CameraPresets::FSTOP_PRESETS[i].f_number;
            const float b = CameraPresets::FSTOP_PRESETS[i + 1].f_number;
            if (f_number >= a && f_number <= b) {
                const float t = (std::log(f_number) - std::log(a)) /
                                (std::log(b) - std::log(a));
                const float la = std::log(CameraPresets::FSTOP_PRESETS[i].aperture_value);
                const float lb = std::log(CameraPresets::FSTOP_PRESETS[i + 1].aperture_value);
                return std::exp(la + (lb - la) * t);
            }
        }
        return CameraPresets::FSTOP_PRESETS[n - 1].aperture_value;
    }

    // ★★ Ters yon AYNI tablodan okunur; aciklik f-sayisinde MONOTON AZALIR.
    static float fNumberForAperture(float ap) {
        const int n = (int)CameraPresets::FSTOP_PRESET_COUNT;
        if (n < 3 || ap <= 1e-6f) return 16.0f;
        if (ap >= CameraPresets::FSTOP_PRESETS[1].aperture_value)
            return CameraPresets::FSTOP_PRESETS[1].f_number;
        if (ap <= CameraPresets::FSTOP_PRESETS[n - 1].aperture_value)
            return CameraPresets::FSTOP_PRESETS[n - 1].f_number;
        for (int i = 1; i < n - 1; ++i) {
            const float a = CameraPresets::FSTOP_PRESETS[i].aperture_value;
            const float b = CameraPresets::FSTOP_PRESETS[i + 1].aperture_value;
            if (ap <= a && ap >= b) {
                const float t = (std::log(ap) - std::log(a)) / (std::log(b) - std::log(a));
                const float la = std::log(CameraPresets::FSTOP_PRESETS[i].f_number);
                const float lb = std::log(CameraPresets::FSTOP_PRESETS[i + 1].f_number);
                return std::exp(la + (lb - la) * t);
            }
        }
        return CameraPresets::FSTOP_PRESETS[n - 1].f_number;
    }

    // ★★★ TEK OKUYUCU. Preset seciliyse tablo degeri, degilse aciklidan
    //   turetilir -- pozlama ile panelin ayni sayiyi gormesinin sarti budur.
    float fNumber() const {
        if (fstop_preset_index > 0 &&
            fstop_preset_index < (int)CameraPresets::FSTOP_PRESET_COUNT)
            return CameraPresets::FSTOP_PRESETS[fstop_preset_index].f_number;
        if (aperture > 1e-5f) return fNumberForAperture(aperture);
        return 16.0f;
    }

    // ★★★ TEK YAZAR: f-sayisini degistiren HER yuzey (HUD ucgeni, kamera
    //   paneli, preset combo'su, `camera.set_fstop_preset`) buradan gecer,
    //   boylece aciklik ile preset indeksi asla ayrismaz. Preset'e oturuyorsa
    //   indeks de oturur; oturmazsa "Custom" (0) yazilir.
    //   ★ ANAHTARA DOKUNMAZ: f-sayisi bulanikligin MIKTARIDIR, VARLIGI degil.
    void setFNumber(float f_number) {
        if (f_number < 0.5f) f_number = 0.5f;
        if (f_number > 128.0f) f_number = 128.0f;
        aperture = apertureForFNumber(f_number);
        lens_radius = aperture * 0.5f;
        fstop_preset_index = 0;
        for (int i = 1; i < (int)CameraPresets::FSTOP_PRESET_COUNT; ++i) {
            if (std::abs(CameraPresets::FSTOP_PRESETS[i].f_number - f_number) < 0.01f) {
                fstop_preset_index = i;
                break;
            }
        }
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // PROFESSIONAL EXPOSURE SETTINGS
    // ═══════════════════════════════════════════════════════════════════════════
    int iso = 100;                     // Current ISO value
    float shutter_speed = 250.0f;      // Shutter speed as 1/x (e.g., 250 = 1/250s)
    int iso_preset_index = 1;          // Default: ISO 100
    int shutter_preset_index = 1;      // Default: 1/4000s
    int fstop_preset_index = 4;        // Default: f/2.8
    int lens_preset_index = 0;         // Default: Custom/Manual
    int body_preset_index = 1;         // Default: Generic Full Frame
    bool auto_exposure = true;         // Default to manual to use above settings
    float ev_compensation = 0.0f;      // EV compensation (-2 to +2)
    // ★ `calculated_ev` 2026-09-06'da SOKULDU: tek yazani ve tek okuyani
    //   hierarchy panelindeki bir satirdi, formulu f-sayisi yerine DoF
    //   `aperture`ini kullaniyordu ve degeri hicbir yerde uygulanmiyordu --
    //   "Exposure" etiketiyle gosterilen olculmemis bir sayiydi. Uygulanan
    //   carpan: `g_display_post.camera_exposure` (post modunun kapisindan
    //   gecmis) veya ham kamera terimi icin `Camera::exposureFactor()`.

    // ═══════════════════════════════════════════════════════════════════════
    // POZLAMA CARPANI -- TEK TANIM
    // ═══════════════════════════════════════════════════════════════════════
    // ★★★★ Bu formul 2026-09-03'e kadar DORT yerde kopyalanmisti:
    //   Main.cpp'de iki kez (CPU denoised preview + render dongusu),
    //   VulkanBackend::setCamera ve OptixBackend. Dordu de ayni sabitleri
    //   (baseline 0.00003125, `* 2.0f`) elle tasiyordu; birini kalibre edip
    //   otekini birakmak, ayni sahnenin iki yolda farkli parlakligi demekti
    //   ve belirtisi "biraz farkli gorunuyor" olurdu.
    //
    // ★★★ Model GORELIDIR ve oyle kalmali: carpan `current_val / baseline_val`
    //   oranidir, mutlak fotometrik formul DEGIL. Ders kitabi formulu
    //   (1 / (1.2 * 2^EV100)) sahne radyansinin cd/m2 olmasini varsayar; bu
    //   motorda isik siddeti keyfi birimde ve mutlak formul her sahneyi
    //   karartir. Buradaki baseline zaten "siyah viewport'u onlemek icin"
    //   kalibre edilmis (asagidaki yorum orijinaldir).
    // ★★★ Oncelik sirasindan GECMIS deger. Kameranin kendi bayraklarini okur;
    //   yeni post yapisinda ekrana giden carpan bu DEGILDIR (bkz.
    //   `rtpost::syncDisplay` ve `g_display_post.camera_exposure`) -- mod
    //   Physical Camera degilse kamera terimi 1.0'dir.
    float exposureFactor() const {
        const float ev_comp = std::pow(2.0f, ev_compensation);
        if (auto_exposure) return ev_comp;
        if (!use_physical_exposure) return ev_comp;
        return physicalExposureFactor();
    }

    // ★★ Bayraklardan BAGIMSIZ fiziksel terim (ISO x enstantane / f^2).
    //   Post zinciri bunu dogrudan cagirir; eskiden ayni sonuc icin Camera'nin
    //   TAM KOPYASI cikarilip iki bayrak zorlaniyordu -- her karede, ve
    //   `nodeName` uzunsa her karede bir tahsisle.
    float physicalExposureFactor() const {
        const float ev_comp = std::pow(2.0f, ev_compensation);

        float iso_mult = 1.0f;
        if (iso_preset_index >= 0 &&
            iso_preset_index < (int)CameraPresets::ISO_PRESET_COUNT) {
            iso_mult = CameraPresets::ISO_PRESETS[iso_preset_index].exposure_multiplier;
        }
        float shutter_time = 0.004f;
        if (shutter_preset_index >= 0 &&
            shutter_preset_index < (int)CameraPresets::SHUTTER_SPEED_PRESET_COUNT) {
            shutter_time = CameraPresets::SHUTTER_SPEED_PRESETS[shutter_preset_index].speed_seconds;
        }
        // ★★★ f-sayisinin TEK tanimi `fNumber()`. Buradaki eski geri dusus
        //   (`0.8f / aperture`) panelin yazma formulunun tersi DEGILDI; artik
        //   ikisi ayni fonksiyondan geliyor. Custom f-stop'lu sahnelerde
        //   pozlama bu yuzden degisir -- beklenen ve DUZELTILMIS davranistir.
        const float f_num = fNumber();

        const float aperture_sq = f_num * f_num;
        const float current_val = (iso_mult * shutter_time) / (aperture_sq + 1e-6f);
        // Calibration: boosted baseline to avoid a black viewport.
        const float baseline_val = 0.00003125f;
        return (current_val / baseline_val) * ev_comp * 2.0f;
    }
    
    // Aspect Ratio for Output (syncs with final render)
    int output_aspect_index = 2;       // Default: 16:9 (index into CameraPresets::ASPECT_RATIOS)

    // PHYSICAL LENS SETTINGS
    // ═══════════════════════════════════════════════════════════════════════════
    bool use_physical_lens = false;    // Toggle between basic FOV and Physical Lens
    float focal_length_mm = 50.0f;     // Focal length in mm (e.g. 24, 35, 50, 85)
    float sensor_width_mm = 36.0f;     // Sensor width (Full Frame = 36mm)
    float sensor_height_mm = 24.0f;    // Sensor height (Full Frame = 24mm)
    bool enable_motion_blur = false;   // Enable Camera Motion Blur (requires velocity calculation)
    float distortion = 0.0f;           // Lens Distortion (-0.5 to 0.5): Negative=Barrel, Positive=Pincushion
    
    // ═══════════════════════════════════════════════════════════════════════════
    // CAMERA MODE - Auto/Pro/Cinema
    // ═══════════════════════════════════════════════════════════════════════════
    CameraMode camera_mode = CameraMode::Pro;
    
    // ═══════════════════════════════════════════════════════════════════════════
    // CINEMA MODE - Lens Imperfections (only active when camera_mode == Cinema)
    // ═══════════════════════════════════════════════════════════════════════════
    
    // Lens Quality (affects all optical aberrations)
    // 0.0 = Vintage/Budget lens (more aberrations)
    // 1.0 = Perfect optical design (minimal aberrations)
    float lens_quality = 0.7f;
    
    // Auto-calculate lens characteristics (true = physics-based, false = manual)
    bool auto_lens_characteristics = false;
    
    // Chromatic Aberration (Renk Sapması)
    bool enable_chromatic_aberration = false;
    float chromatic_aberration = 0.0f;      // 0-1: Lateral CA amount
    float chromatic_aberration_r = 1.002f;  // Red channel offset multiplier
    float chromatic_aberration_b = 0.998f;  // Blue channel offset multiplier
    
    // Vignetting (Köşe Kararması)
    bool enable_vignetting = false;
    float vignetting_amount = 0.0f;         // 0-1: Vignette strength
    float vignetting_falloff = 2.0f;        // Falloff curve exponent (1.5-4.0)
    
    // Calculate lens characteristics from physical properties
    // Call this when focal_length, aperture, or lens_quality changes
    void calculateLensCharacteristics() {
        if (!auto_lens_characteristics || camera_mode != CameraMode::Cinema) return;
        
        // Get current f-stop
        float f_number = 2.8f;
        if (fstop_preset_index > 0 && fstop_preset_index < 12) {
            const float fstops[] = {0, 1.2f, 1.4f, 1.8f, 2.0f, 2.8f, 4.0f, 5.6f, 8.0f, 11.0f, 16.0f, 22.0f};
            f_number = fstops[fstop_preset_index];
        }
        
        // ─────────────────────────────────────────────────────────────────────
        // VIGNETTING: Based on focal length and aperture
        // - Wide angle = more mechanical vignetting
        // - Wide aperture = more optical vignetting
        // - Stopping down reduces vignetting significantly
        // ─────────────────────────────────────────────────────────────────────
        float focal_factor = 1.0f;
        if (focal_length_mm < 24.0f) focal_factor = 1.5f;       // Ultra wide
        else if (focal_length_mm < 35.0f) focal_factor = 1.2f;  // Wide
        else if (focal_length_mm < 50.0f) focal_factor = 1.0f;  // Normal
        else if (focal_length_mm < 85.0f) focal_factor = 0.8f;  // Portrait
        else focal_factor = 0.6f;                                // Telephoto
        
        // Aperture effect: wide open = more vignetting
        float aperture_vignette = 1.0f / (f_number * 0.5f);     // f/1.4 = 1.43, f/8 = 0.25
        aperture_vignette = (std::min)(aperture_vignette, 1.0f);
        
        // Quality reduces vignetting
        float quality_reduction = 1.0f - (lens_quality * 0.6f);
        
        vignetting_amount = focal_factor * aperture_vignette * quality_reduction * 0.4f;
        vignetting_amount = std::clamp(vignetting_amount, 0.0f, 0.8f);
        vignetting_falloff = 2.0f + (1.0f - lens_quality);
        enable_vignetting = (vignetting_amount > 0.02f);
        
        // ─────────────────────────────────────────────────────────────────────
        // CHROMATIC ABERRATION: Based on lens quality and aperture
        // - Budget lenses have more CA
        // - Wide aperture = more visible CA
        // - Stopping down reduces CA
        // ─────────────────────────────────────────────────────────────────────
        float ca_base = (1.0f - lens_quality) * 0.02f;  // 0 to 0.02 based on quality
        
        // Aperture effect: wide open = more CA
        float aperture_ca = (2.8f / f_number);  // f/1.4 = 2, f/8 = 0.35
        aperture_ca = std::clamp(aperture_ca, 0.2f, 2.0f);
        
        chromatic_aberration = ca_base * aperture_ca;
        chromatic_aberration = std::clamp(chromatic_aberration, 0.0f, 0.03f);
        
        // R/B channel scales
        chromatic_aberration_r = 1.0f + chromatic_aberration * 0.5f;  // Red bends outward
        chromatic_aberration_b = 1.0f - chromatic_aberration * 0.5f;  // Blue bends inward
        
        enable_chromatic_aberration = (chromatic_aberration > 0.001f);
        
        // ─────────────────────────────────────────────────────────────────────
        // AUTO DISTORTION: Based on focal length
        // ─────────────────────────────────────────────────────────────────────
        if (focal_length_mm < 24.0f) {
            distortion = -0.15f * (1.0f - lens_quality);  // Barrel
        } else if (focal_length_mm > 100.0f) {
            distortion = 0.05f * (1.0f - lens_quality);   // Pincushion
        } else {
            distortion = 0.0f;  // Normal range - minimal distortion
        }
    }
    
    // Focus Breathing (Odak Soluması - FOV changes with focus)
    bool enable_focus_breathing = false;
    float focus_breathing_amount = 0.05f;   // % FOV change per focus distance change
    
    // Lens Flare
    bool enable_lens_flare = false;
    float lens_flare_intensity = 0.5f;
    float lens_flare_threshold = 0.9f;      // Brightness threshold to trigger flare
    bool anamorphic_flare = false;          // Horizontal blue streak (cinema style)
    
    
    // ═══════════════════════════════════════════════════════════════════════════
    // CAMERA SHAKE / HANDHELD SIMULATION
    // ═══════════════════════════════════════════════════════════════════════════
    bool enable_camera_shake = false;
    float shake_intensity = 0.03f;          // Overall shake multiplier (0-1), 0.03 = Professional default
    float shake_frequency = 8.0f;           // Hz (hand tremor ~8-12Hz)
    
    // Handheld physics
    float handheld_sway_amplitude = 0.005f;   // Body sway (meters)
    float handheld_sway_frequency = 0.5f;     // Hz
    float breathing_amplitude = 0.003f;       // Breathing motion (meters)
    float breathing_frequency = 0.25f;        // ~15 breaths/minute
    
    // Focus Drift (shake-induced focus variation)
    bool enable_focus_drift = true;           // Focus follows shake movement
    float focus_drift_amount = 0.1f;          // Max focus distance variation (meters)
    
    // Operator skill (affects shake reduction)
    enum class OperatorSkill { Amateur, Intermediate, Professional, Expert };
    OperatorSkill operator_skill = OperatorSkill::Professional;
    
    // IBIS (In-Body Image Stabilization)
    bool ibis_enabled = false;
    float ibis_effectiveness = 5.0f;        // Stops of stabilization (typically 3-8 stops)
    
    // ═══════════════════════════════════════════════════════════════════════════
    // PHYSICAL EXPOSURE (Fiziksel Pozlama)
    // ═══════════════════════════════════════════════════════════════════════════
    int native_iso = 100;                   // Sensor native ISO (for noise calculation)
    bool use_physical_exposure = false;     // Use physical exposure calculation
    
    // Shutter Angle (Cinema style) - Alternative to shutter speed
    bool use_shutter_angle = false;
    float shutter_angle = 180.0f;           // Degrees (180 = 50% duty cycle)
    
    // Get physical exposure multiplier
    // This multiplies the render result (signal + variance together!)
    float getPhysicalExposureMultiplier() const {
        if (!use_physical_exposure) {
            return std::pow(2.0f, ev_compensation);
        }
        
        // Return 0.0 to signal the backend to use its own physical presets
        // Or return a baseline if you want a fallback
        return 0.0f; 
    }
    
    // Get recommended sample count based on ISO
    // Higher ISO = more samples needed to reduce visible variance
    int getRecommendedSamples(int base_samples = 64) const {
        if (iso <= native_iso) return base_samples;
        
        float iso_factor = std::log2(static_cast<float>(iso) / 100.0f);
        return base_samples * static_cast<int>(std::pow(2.0f, (std::max)(0.0f, iso_factor * 0.5f)));
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // CAMERA RIG SYSTEM (Dolly, Crane, Orbit, Handheld, Steadicam)
    // ═══════════════════════════════════════════════════════════════════════════
    enum class RigMode { Static, Dolly, Crane, Orbit, Handheld, Steadicam };
    RigMode rig_mode = RigMode::Static;
    
    // Dolly - Linear track movement
    float dolly_position = 0.0f;       // Position along track (units)
    float dolly_speed = 1.0f;          // Movement speed multiplier
    Vec3 dolly_start_pos;              // Initial position when dolly started
    Vec3 dolly_end_pos;                // End position for dolly track
    
    // Crane - Arm with boom
    float crane_arm = 5.0f;            // Arm length
    float crane_height = 2.0f;         // Base height
    float crane_boom = 0.0f;           // Boom angle (-45 to +45)
    
    // Orbit - Around target
    float orbit_angle = 0.0f;          // Current angle
    float orbit_radius = 5.0f;         // Distance from target
    Vec3 orbit_target;                 // Point to orbit around
    
    // Steadicam - Smoothed movement
    float steadicam_smoothing = 0.9f;  // Position smoothing (0-1)
    
    // Camera physics (for realistic motion)
    float camera_mass_kg = 1.3f;       // Body + lens mass
    float camera_damping = 5.0f;       // Movement damping
    Vec3 camera_velocity;              // Current velocity (m/s)
    Vec3 camera_angular_velocity;      // Current angular velocity (rad/s)
    
    // ═══════════════════════════════════════════════════════════════════════════
    // STATE MANAGEMENT (Dirty Flag Architecture)
    // ═══════════════════════════════════════════════════════════════════════════
    bool is_dirty = false;
    
    void markDirty() {
        is_dirty = true;
    }

    bool checkDirty() {
        bool was_dirty = is_dirty;
        is_dirty = false;
        return was_dirty;
    }

private:
    // Initial state for reset
    Vec3 init_lookfrom;
    Vec3 init_lookat;
    Vec3 init_vup;
    float init_vfov = 45.0f;
    float init_aperture = 0.0f;
    float init_focus_dist = 10.0f;
    void updateFrustumPlanes();

    Vec3 getViewDirection() const;

    // Frustum culling i�in ek alanlar

    Plane frustum_planes[6];
};

#endif // CAMERA_H




