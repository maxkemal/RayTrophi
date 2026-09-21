#pragma once
#include "Api/RtApiRayFusion.h"
#include "Api/RtApi.h"
#include "imgui.h"
#include "UI/DiagnosticsLayout.h"
#include "UI/ScreenGiUI.h"
#include "UI/ReflectionUI.h"

// ═══════════════════════════════════════════════════════════════════════════
// RAYFUSION TANILAMA — alt sistem başına bir grup
// ═══════════════════════════════════════════════════════════════════════════
//
// ★★★★ DÜZEN KURALI: her grup BİR alt sistemdir ve kendi kontrollerini kendi
//   sayılarının yanında taşır. Yedi alt sistemin anahtarını tek bir "Controls"
//   yığınına toplamak da aynı dağınıklıktır: bir kadranı çevirdikten sonra
//   sonucunu görmek için başka bir bölüme gitmek gerekirdi.
//
// ★★★ Grup başlığındaki NOKTA, bölüm kapalıyken de sağlığı söyler. Kapalı bir
//   bölümün "sorun yok" anlamına gelmesi, tanılamanın en sessiz kaybıdır.
//
// ★★ Düzyazı `RtDiag::Help()` içine (hover) taşındı. Panelin gövdesinde artık
//   yalnızca ölçüm satırları var; gerekçeler kod yorumunda ve
//   docs/dev/NEXT_BUILD_CHECKS.md içinde yaşar.
//
// ★ "Yapamaz" ile "henüz yapmadı" AYNI gösterilemez: donanım RT yoksa ışına
//   dayanan her kontrol ImGui::BeginDisabled ile ÇEKİRDEKTEN gelen gerekçeye
//   sarılır. Buraya yazılmış sabit bir metin, çekirdek değişince yalancı olurdu.
inline void DrawRayFusionDevelopmentStatus() {
    if (!ImGui::TreeNode("RayFusion diagnostics##CoreStatus")) return;

    const auto status  = rtapi::rayFusionCoreStatus();
    const auto sceneAs = rtapi::rayFusionSceneASStatus();
    const auto probes  = rtapi::rayFusionProbeFieldStatus();
    const auto shadow  = rtapi::rtShadow();
    const bool hwRt    = sceneAs.hardware_rt;

    // Kopya tamponu HER karede sifirlanir ve satirlar cizilirken dolar.
    RtDiag::ResetBuffer("RayFusion diagnostics");
    RtDiag::CopyAllButton("Copy all##RayFusionDiag");
    ImGui::SameLine();
    ImGui::TextDisabled("hover (?) for reasoning");

    if (!hwRt) {
        RtDiag::Fault("hardware ray tracing unavailable on this device");
        RtDiag::Note("%s", sceneAs.inactive_reason.empty()
            ? "the device reported no ray tracing support"
            : sceneAs.inactive_reason.c_str());
        RtDiag::Help("Indirect lighting needs probe rays, so the RayFusion GI steps stay off "
                     "here. The prefiltered sky ambient path is unaffected and keeps working.");
    }
    ImGui::Separator();

    // -- 1. Ekran-uzayi difuz GI --------------------------------------------
    DrawScreenGiGroup();

    // -- 2. Yansimalar ------------------------------------------------------
    // ReflectionUI.h bu partinin disinda; kendi cizimini koruyup yalnizca
    // gruba sariyoruz. Sarmalamak, sahibi olmadigimiz bir paneli yeniden
    // yazmaktan daha guvenli ve duzeni yine de tek tiplestiriyor.
    if (RtDiag::BeginGroup("##ReflectionGroup", "Reflections", RtDiag::State::Ok, nullptr)) {
        DrawReflectionControls();
        RtDiag::EndGroup();
    }

    // -- 3. Isinla izlenen golgeler -----------------------------------------
    {
        RtDiag::State st = RtDiag::State::Off;
        const char* txt = "off";
        if (shadow.enabled) {
            st = shadow.ready ? RtDiag::State::Ok : RtDiag::State::Warn;
            txt = shadow.ready ? "active" : "requested";
        }
        if (RtDiag::BeginGroup("##RtShadowGroup", "Ray-traced shadows", st, txt)) {
            bool enabled = shadow.enabled;
            static std::string err;
            if (!shadow.supported && !shadow.enabled) ImGui::BeginDisabled();
            if (ImGui::Checkbox("Enabled##RayFusionShadow", &enabled)) {
                const auto r = rtapi::setRtShadow(enabled);
                err = r.ok ? std::string{} : r.error;
            }
            if (!shadow.supported && !shadow.enabled) ImGui::EndDisabled();
            RtDiag::Help("Hard shadows for one directional light or the world sun. "
                         "Other lights use cascades.");
            if (shadow.enabled && shadow.ready) RtDiag::Row("budget", "%u rays/frame", shadow.rays);
            else if (!shadow.reason.empty()) RtDiag::Note("%s", shadow.reason.c_str());
            if (!err.empty()) RtDiag::Fault("%s", err.c_str());
            RtDiag::EndGroup();
        }
    }

    // -- 4. Probe alani -----------------------------------------------------
    bool followCamera = probes.follow_camera;
    {
        RtDiag::State st = probes.supported
            ? (probes.valid > 0u ? RtDiag::State::Ok : RtDiag::State::Warn)
            : RtDiag::State::Off;
        char head[64];
        snprintf(head, sizeof(head), "%u/%u valid", probes.valid, probes.total);
        if (RtDiag::BeginGroup("##ProbeGroup", "Probe field", st, head)) {
            bool overlay = probes.overlay_requested;
            bool traced  = probes.producer_traced_requested;
            if (!probes.supported) ImGui::BeginDisabled();
            if (ImGui::Checkbox("Show probes##RayFusion", &overlay))
                rtapi::setRayFusionProbeOverlay(overlay);
            RtDiag::Help("Green: usable | Amber: pending | Red: rejected inside geometry. "
                         "Markers behind scene depth are hidden; Rendered mode draws none.");
            if (ImGui::Checkbox("Follow camera##RayFusion", &followCamera))
                rtapi::setRayFusionProbeFollowCamera(followCamera);
            RtDiag::Help("Fixed probe/ray budget. Camera translation scrolls new cells; "
                         "disabling freezes the current window.");
            if (!probes.supported) ImGui::EndDisabled();

            // Uretici secici: A/B OLCUM kolu. Kapatilamayan bir uretici,
            // yerini aldigi ureticiye karsi OLCULEMEZ.
            if (!hwRt) ImGui::BeginDisabled();
            if (ImGui::Checkbox("Trace probe rays (1b)##RayFusion", &traced))
                rtapi::setRayFusionProbeProducer(traced);
            if (!hwRt) ImGui::EndDisabled();
            if (!hwRt && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                ImGui::SetTooltip("%s", sceneAs.inactive_reason.empty()
                    ? "This device reports no hardware ray tracing."
                    : sceneAs.inactive_reason.c_str());

            RtDiag::Row("producer", "%s", probes.producer.c_str());
            // ISTENEN ile CALISAN ayri: istegi yankilayan bir gosterge
            // reddedilmis bir istegi asla gosteremez.
            if (hwRt && traced && probes.producer != "traced") {
                RtDiag::Warn("requested tracing, running '%s'", probes.producer.c_str());
                if (!probes.producer_reason.empty())
                    RtDiag::Note("%s", probes.producer_reason.c_str());
            }
            RtDiag::Row("slots", "%u valid / %u total / %u pending",
                        probes.valid, probes.total, probes.pending);
            // Hangi rejimin kostugu, ne kadar kaba oldugundan DAHA bilgilendirici:
            // ayni kaba izgara iki farkli sebepten dogar ve duzeltmeleri farklidir.
            RtDiag::Row("placement", "%s%s", probes.auto_fit ? "auto: " : "manual",
                        probes.auto_fit ? probes.auto_fit_mode.c_str() : "");
            if (!probes.auto_fit_reason.empty())
                RtDiag::Note("%s", probes.auto_fit_reason.c_str());
            RtDiag::Row("budget", "%s, %llu rejected", probes.budget_preset.c_str(),
                        static_cast<unsigned long long>(probes.rejected));
            if (probes.producer == "traced") {
                RtDiag::Row("rays hitting geometry", "%.0f%%   trace %.2f ms",
                            probes.hit_fraction * 100.0f, probes.trace_ms);
                RtDiag::Help("If this stays at zero the probes see sky in every direction and "
                             "the image is byte-identical to step 1a -- a nothing that looks "
                             "correct. It is the first number to check.");
                if (probes.hit_fraction <= 0.0f)
                    RtDiag::Warn("no ray hit anything: indistinguishable from step 1a");
                RtDiag::Row("mean hit distance", "%.2f (cell %.2f)",
                            probes.mean_hit_distance, probes.spacing);
                if (probes.rejected_inside > 0u)
                    RtDiag::Note("%u probe(s) born inside geometry, published as 'not measured'",
                                 probes.rejected_inside);
                // ★ Bu satir duzeltmenin KANITI: "kirmizi probe kalmadi" hem
                //   "kapi calisti" hem "hic probe yayinlanmadi" olabilir.
                if (probes.backface_enclosed > 0u)
                    RtDiag::Note("%u probe(s) saw mostly back faces but are in OPEN space "
                                 "(interior room, not inside a solid)", probes.backface_enclosed);
                RtDiag::Help("Back-face fraction alone cannot tell 'inside a solid' from 'inside "
                             "a closed room whose wall normals point outward' -- both look "
                             "identical. Distance separates them: geometry within a quarter cell "
                             "means embedded, metres away means a room.");
            }
            RtDiag::Row("planned", "%u rays, %u probes / update",
                        status.planned_rays_per_update, status.planned_probes_per_update);
    if (probes.configured && ImGui::TreeNode("Probe window##RayFusionGrid")) {
        // Duzenleme tamponu UYGULANMIS pencereden tohumlanir. Kendi basina
        // yasayan bir tampon, kamera takibi pencereyi kaydirdiktan sonra
        // ekranda eski sayilari tutar ve panel yalan soylemeye baslar.
        static int editCounts[3] = {0, 0, 0};
        static int editMinimum[3] = {0, 0, 0};
        static float editSpacing = 0.0f;
        static bool dirty = false;
        static uint64_t seededFrom = ~0ull;
        const uint64_t appliedKey =
            (static_cast<uint64_t>(probes.counts[0]) << 40) ^
            (static_cast<uint64_t>(probes.counts[1]) << 32) ^
            (static_cast<uint64_t>(probes.counts[2]) << 24) ^
            (static_cast<uint64_t>(static_cast<uint32_t>(probes.minimum[0])) << 16) ^
            (static_cast<uint64_t>(static_cast<uint32_t>(probes.minimum[1])) << 8) ^
            static_cast<uint64_t>(static_cast<uint32_t>(probes.minimum[2])) ^
            (static_cast<uint64_t>(probes.spacing * 1000.0f) << 48);
        if (seededFrom != appliedKey && !dirty) {
            for (int axis = 0; axis < 3; ++axis) {
                editCounts[axis] = static_cast<int>(probes.counts[axis]);
                editMinimum[axis] = probes.minimum[axis];
            }
            editSpacing = probes.spacing;
            seededFrom = appliedKey;
        }
        ImGui::SetNextItemWidth(RtDiag::labelWidth());
        dirty |= ImGui::InputInt3("Cells (x,y,z)##RayFusionGrid", editCounts);
        ImGui::SetNextItemWidth(RtDiag::labelWidth());
        dirty |= ImGui::InputFloat("Cell size##RayFusionGrid", &editSpacing, 0.1f, 1.0f, "%.3f");
        if (followCamera) ImGui::BeginDisabled();
        ImGui::SetNextItemWidth(RtDiag::labelWidth());
        dirty |= ImGui::InputInt3("Lowest cell##RayFusionGrid", editMinimum);
        if (followCamera) ImGui::EndDisabled();
        RtDiag::Help("Changing the cell counts or the cell size drops every measurement: the "
                     "cells cover different world volumes. Moving the window keeps the cells "
                     "that stay inside it. The applied window appears only after the viewport "
                     "services the field on its next frame. While camera following is on it "
                     "owns the placement.");

        const int slots = editCounts[0] * editCounts[1] * editCounts[2];
        RtDiag::Row("slots", "%d of %u", slots, probes.max_slots);
        if (editSpacing > 0.0f)
            RtDiag::Note("extent %.1f x %.1f x %.1f from (%.1f, %.1f, %.1f)",
                         editCounts[0] * editSpacing, editCounts[1] * editSpacing,
                         editCounts[2] * editSpacing, editMinimum[0] * editSpacing,
                         editMinimum[1] * editSpacing, editMinimum[2] * editSpacing);
        static std::string gridError;
        const bool overBudget = slots > static_cast<int>(probes.max_slots);
        if (overBudget) RtDiag::Fault("over the slot ceiling; reduce the cell counts");
        if (overBudget || !dirty) ImGui::BeginDisabled();
        if (ImGui::Button("Apply##RayFusionGrid")) {
            RayFusion::GridRequest request;
            request.hasCounts = true;
            for (int axis = 0; axis < 3; ++axis)
                request.counts[axis] = static_cast<uint32_t>(editCounts[axis] < 1 ? 1 : editCounts[axis]);
            request.hasSpacing = true;
            request.spacing = editSpacing;
            if (!followCamera) {
                request.hasMinimum = true;
                for (int axis = 0; axis < 3; ++axis) request.minimum[axis] = editMinimum[axis];
            }
            gridError.clear();
            if (rtapi::setRayFusionProbeGrid(request, gridError)) {
                dirty = false;
                seededFrom = ~0ull;  // reseed from whatever actually got applied
            }
        }
        if (overBudget || !dirty) ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::Button("Revert##RayFusionGrid")) {
            dirty = false; seededFrom = ~0ull; gridError.clear();
        }
        if (!gridError.empty()) RtDiag::Fault("%s", gridError.c_str());
        ImGui::TreePop();
    }

            RtDiag::EndGroup();
        }
    }

    // -- 5. Tek difuz sicrama (1b-beta) -------------------------------------
    {
        const auto& b = probes.bounce;
        RtDiag::State st = RtDiag::State::Off;
        const char* txt = "off";
        if (b.requested) {
            st = (b.ready && b.shadedHits > 0u) ? RtDiag::State::Ok : RtDiag::State::Warn;
            txt = b.active ? "active" : "requested";
        }
        if (RtDiag::BeginGroup("##BounceGroup", "Diffuse bounce (1b-beta)", st, txt)) {
            bool bounce = b.requested;
            if (!hwRt) ImGui::BeginDisabled();
            if (ImGui::Checkbox("Enabled##RayFusion", &bounce))
                rtapi::setRayFusionProbeBounce(bounce);
            if (!hwRt) ImGui::EndDisabled();
            RtDiag::Help("Bounce off: visibility-only probes (1b-alpha). Bounce on: one diffuse "
                         "surface bounce for supported materials. Trace probe rays must also be "
                         "enabled. Unsupported materials remain opaque occluders.");

            // Her iki sayi da "N of TOTAL usable". Daha once REDDEDILEN sayi
            // yaziliyordu ve "hicbir isik desteklenmiyor" diye okunuyordu --
            // cumlesi tersten okunabilen bir sayac olcum degildir.
            RtDiag::Row("materials", "%u of %u supported", b.supportedMaterials, b.materials);
            RtDiag::Row("lights", "%u of %u usable%s", b.lights,
                        b.lights + b.unsupportedLights,
                        b.sunInBounce ? " (+ sky sun)" : "");
            if (b.sunInBounce && !b.sunTintFromLut)
                RtDiag::Warn("sun tint is the constant fallback, not the transmittance LUT");
            if (!b.sunInBounce)
                RtDiag::Note("Physical Sky sun is NOT carried indirectly");
            RtDiag::Help("Without the sun in this table the bounce can only carry sky light, so "
                         "an interior lit through a window goes blue instead of taking the warm "
                         "bounce off the sunlit floor. Measured against the RT reference, that "
                         "was the whole remaining difference.");
            if (b.unsupportedMaterials > 0u) {
                RtDiag::Note("rejected by: textured %u, transparent %u, layered %u, flags %u (0x%X)",
                             b.rejectedTextured, b.rejectedTransparent,
                             b.rejectedLayered, b.rejectedFlagged, b.rejectedFlagBits);
                RtDiag::Help("The clause that rejected each material is what can be compared "
                             "against the material panel. A bare '31 unsupported' is the correct "
                             "answer in a fully textured scene AND the same number with a broken "
                             "gate, so the count alone is not a measurement.");
            }
            if (b.requested && b.ready && b.supportedMaterials == 0u && b.materials > 0u)
                RtDiag::Warn("no material qualifies: the bounce cannot change this image");

            if (b.requested && b.ready) {
                RtDiag::Row("last dispatch", "%u hits, %u shaded", b.hits, b.shadedHits);
                RtDiag::Help("Everything above describes what the slice COULD shade. This is "
                             "what it DID shade. Rays hitting geometry while nothing gets shaded "
                             "is the failure that looks exactly like success.");
                if (b.hits > 0u && b.shadedHits == 0u)
                    RtDiag::Warn("rays hit geometry but NOTHING was shaded: equals 1b-alpha");
                if (b.hits > 0u) {
                    // backFaceShaded toplamda YOK: golgelenen isabetlerin
                    // bir alt kumesi, ayri bir cikis yolu degil.
                    const uint32_t named = b.shadedHits +
                        b.skippedBounceDisabled + b.rejectedUnresolved +
                        b.rejectedUnsupported + b.rejectedDegenerate;
                    RtDiag::Note("not shaded: bounce-off %u, unresolved %u, "
                                 "unsupported %u, degenerate %u",
                                 b.skippedBounceDisabled, b.rejectedUnresolved,
                                 b.rejectedUnsupported, b.rejectedDegenerate);
                    RtDiag::Note("of the shaded, %u were back faces (normal flipped)",
                                 b.backFaceShaded);
                    if (named != b.hits)
                        RtDiag::Fault("%u hits unaccounted for: an exit path has no counter",
                                      b.hits > named ? b.hits - named : named - b.hits);
                    // Artik bir uyari DEGIL: arka yuzler golgeleniyor. Yuksek
                    // oran yalnizca sahnenin ic mekan oldugunu soyler.
                    if (b.backFaceShaded > b.hits / 2u)
                        RtDiag::Note("interior scene: most hits are back faces, shaded via "
                                     "flipped normals like the RT reference");
                }
                if (b.alphaTested > 0u)
                    RtDiag::Note("alpha-tested %u, of which %u occluded",
                                 b.alphaTested, b.alphaOccluded);
            }
            RtDiag::Row("table prepare", "%.2f ms/frame (CPU)", b.prepareMs);
            RtDiag::Note("instances %.2f | emissive %.2f%s | materials %.2f | upload %.2f%s",
                         b.prepareInstancesMs, b.prepareEmissiveMs,
                         b.emissiveCached ? " (cached)" : " (SCANNED)",
                         b.prepareMaterialsMs, b.prepareUploadMs,
                         b.uploadSkipped ? " (skipped)" : "");
            RtDiag::Help("Paid on EVERY raster frame, before any ray is cast -- also while the "
                         "bounce is off. Unmeasured it looks like a few percent CPU, which on 16 "
                         "logical cores is one core at 100 percent.");
            if (b.prepareMs > 2.0)
                RtDiag::Warn("paid on every raster frame, before any ray is cast");
            if (!b.reason.empty()) RtDiag::Note("%s", b.reason.c_str());

            // Emissive NEE: kullanicinin en cok karsilasacagi surpriz (lamba
            // abajurlari transparan, yani en beklenen emissive nesne tam olarak
            // katki VERMEYEN nesne) -- ama bir tanilama, surekli durmasi gerekmez.
            if ((b.emissiveTriangles > 0u || b.emissiveDropped > 0u ||
                 b.emissiveSkippedIndexed > 0u || b.emissiveRejectedTransparent > 0u) &&
                ImGui::TreeNode("Emissive lights##RayFusionEmissive")) {
                RtDiag::Row("area lights", "%u triangles (%.3g m2)",
                            b.emissiveTriangles, double(b.emissiveArea));
                if (b.emissiveDropped > 0u || b.emissiveSkippedIndexed > 0u ||
                    b.emissiveRejectedTransparent > 0u)
                    RtDiag::Note("excluded: cap %u tri, welded mesh %u, transparent %u material",
                                 b.emissiveDropped, b.emissiveSkippedIndexed,
                                 b.emissiveRejectedTransparent);
                if (b.emissiveRejectedTransparent > 0u)
                    RtDiag::Warn("transparent emissives contribute nothing: a glass lamp shade "
                                 "is outside the bounce subset");
                ImGui::TreePop();
            }
            RtDiag::EndGroup();
        }
    }

    // -- 6. Sahne hizlandirma yapisi ----------------------------------------
    {
        RtDiag::State st = !hwRt ? RtDiag::State::Off
                         : (sceneAs.ready ? RtDiag::State::Ok : RtDiag::State::Warn);
        if (RtDiag::BeginGroup("##SceneAsGroup", "Scene AS", st,
                               !hwRt ? "unavailable" : (sceneAs.ready ? "built" : "not built"))) {
            if (!hwRt) {
                RtDiag::Note("hardware ray tracing required");
            } else if (!sceneAs.ready) {
                RtDiag::Note("%s", sceneAs.inactive_reason.empty()
                    ? "no reason reported" : sceneAs.inactive_reason.c_str());
            } else {
                RtDiag::Row("size", "%u BLAS, %u instances, %.1f MB",
                            sceneAs.blas_count, sceneAs.instance_count,
                            static_cast<double>(sceneAs.as_bytes) / (1024.0 * 1024.0));
                RtDiag::Row("triangulation", "%u indexed, %u flat SoA",
                            sceneAs.blas_indexed, sceneAs.blas_flat);
                RtDiag::Help("Building a welded (indexed) mesh without its index buffer moves no "
                             "counter and traces a surface that does not exist. In a scene with "
                             "terrain or static imports, 'indexed 0' is a fault.");
                RtDiag::Row("builds", "%.2f ms last, %llu builds, %llu TLAS-only",
                            sceneAs.last_build_ms,
                            static_cast<unsigned long long>(sceneAs.builds),
                            static_cast<unsigned long long>(sceneAs.tlas_only_refreshes));
                if (sceneAs.instances_hidden > 0u) {
                    RtDiag::Note("%u hidden instance excluded", sceneAs.instances_hidden);
                    RtDiag::Help("Delete HIDES (mask=0); the BLAS stays resident for undo. "
                                 "'3 BLAS but 2 instances' is therefore correct.");
                }
                if (sceneAs.instances_skipped > 0u)
                    RtDiag::Fault("%u instance over the cap are NOT in the traced scene",
                                  sceneAs.instances_skipped);
                if (sceneAs.meshes_skipped > 0u)
                    RtDiag::Fault("%u mesh produced no BLAS", sceneAs.meshes_skipped);
            }
            RtDiag::EndGroup();
        }
    }

    // -- 7. Cekirdek denetimi -----------------------------------------------
    ImGui::Separator();
    static rtapi::RayFusionCoreValidation lastResult;
    if (ImGui::Button("Validate core##RayFusion")) {
        lastResult = rtapi::validateRayFusionCore();
        ImGui::OpenPopup("Core result##RayFusion");
    }
    RtDiag::Help("Checks cache lifetime and scheduling on the CPU control plane. It does NOT "
                 "test GPU rendering, and it answers on a machine without ray tracing too.");
    if (ImGui::BeginPopup("Core result##RayFusion")) {
        ImGui::Text("%s (%u checks)", lastResult.passed ? "Passed" : "Failed",
                    static_cast<unsigned>(lastResult.checks.size()));
        for (const auto& check : lastResult.checks)
            if (!check.passed) ImGui::TextWrapped("%s: %s", check.name.c_str(), check.detail.c_str());
        ImGui::TextDisabled("GPU rendering was not tested.");
        ImGui::EndPopup();
    }
    ImGui::TreePop();
}
