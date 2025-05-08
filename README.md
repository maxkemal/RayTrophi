# RayTrophi

## v0.02 (Mayıs 2025)

**Türkçe (TR):**

Bu sürümde önemli yapısal güncellemeler ve yeni özellikler eklendi:

- 🔥 **OptiX pipeline güncellendi**: Yeni raygen yapısı ve material scatter GPU tarafında iyileştirildi.
- 🎞 **Animasyon desteği**: Kamera, ışık ve obje animasyonları Assimp üzerinden destekleniyor.
- 🧠 **Principled BSDF güncellemeleri**: CPU ve GPU tarafında daha doğru fiziksel davranış.
- 📷 **GPU kamera sistemi**: Kamera parametreleri GPU'ya aktarılarak dinamik kontrol sağlandı.
- 🧹 **Kod modernizasyonu**: Eski metodlar güncellendi, gereksiz hesaplamalar kaldırıldı.
- ✅ **CPU-Embree-OptiX senkronizasyonu**: Tüm yollar artık benzer materyal ve ışık sonuçları veriyor.

> Not: v0.01 ile uyumsuz değişiklikler içerir. Eski sürümü kullanmak isteyenler `v0.01` commitine dönebilir.

**Gelecek Planları (v0.03 ve sonrası):**
- Volumetrik materyal GPU tarafına taşınacak.
- Multiple Importance Sampling (MIS) iyileştirilecek.
- Bone animasyonları GPU'da desteklenecek.

---

**English (EN):**

This version includes major structural updates and new features:

- 🔥 **OptiX pipeline updated**: New raygen structure and improved material scatter on the GPU side.
- 🎞 **Animation support**: Camera, light, and object animations are now supported via Assimp.
- 🧠 **Principled BSDF updates**: More accurate physical behavior on both CPU and GPU.
- 📷 **GPU camera system**: Camera parameters can now be dynamically controlled on the GPU.
- 🧹 **Code modernization**: Outdated methods were updated, and unnecessary calculations were removed.
- ✅ **CPU-Embree-OptiX synchronization**: All paths now provide consistent material and lighting results.

> Note: Contains breaking changes compared to v0.01. Users who wish to use the previous version can revert to the `v0.01` commit.

**Upcoming Plans (v0.03 and beyond):**
- Volumetric material support will be moved to GPU.
- Multiple Importance Sampling (MIS) improvements.
- Bone animations will be supported on the GPU.

---
