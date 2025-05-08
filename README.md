# RayTrophi

## v0.02 (Mayıs 2025)

Bu sürümde önemli yapısal güncellemeler ve yeni özellikler eklendi:

- 🔥 **OptiX pipeline güncellendi**: Yeni raygen yapısı ve material scatter GPU tarafında iyileştirildi.
- 🎞 **Animasyon desteği**: Kamera, ışık ve obje animasyonları Assimp üzerinden destekleniyor.
- 🧠 **Principled BSDF güncellemeleri**: CPU ve GPU tarafında daha doğru fiziksel davranış.
- 📷 **GPU kamera sistemi**: Kamera parametreleri GPU'ya aktarılarak dinamik kontrol sağlandı.
- 🧹 **Kod modernizasyonu**: Eski metodlar güncellendi, gereksiz hesaplamalar kaldırıldı.
- ✅ **CPU-Embree-OptiX senkronizasyonu**: Tüm yollar artık benzer materyal ve ışık sonuçları veriyor.

> Not: v0.01 ile uyumsuz değişiklikler içerir. Eski sürümü kullanmak isteyenler `v0.01` commitine dönebilir.

---

## Gelecek Planları (v0.03 ve sonrası)
- Volumetrik materyal GPU tarafına taşınacak.
- Multiple Importance Sampling (MIS) iyileştirilecek.
- Bone animasyonları GPU'da desteklenecek.
