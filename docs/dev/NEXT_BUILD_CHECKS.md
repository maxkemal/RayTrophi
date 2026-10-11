# 2026-10-11 parti 5: gizmolar perspektifte oturuyor (derinlik testi)

Önceki parti (gaz envanteri: sanal sıcaklık, yoğuşmayan gaz, rüzgârla yağmur)
canlı PASS ve commit'li (`331cf5c`, `ffd7619`); kanıt BIRLESIK §4.8c.

**Sorun:** sahne gizmoları ImGui 2D çizgileriydi; derinlik taşımadıkları için
her şeyin önünde çiziliyorlardı. Domain kutusu bir duvarın arkasında olsa bile
önünde gibi görünüyordu, bu yüzden konum vermek için üst/ön görünüme geçmek
gerekiyordu.

**Çözüm:** Yeni `UI/GizmoOcclusion.{h,cpp}` modülü.
- Her çizgi ekranda ~6 piksellik parçalara bölünür. Her parçaya gözden, tıklamayla
  seçmenin kullandığı CPU BVH'si üzerinden bir ışın atılır. Bu her backend'de
  çalışır; ışında hacimler yok sayılır.
- Arkada kalan parça soluk çizilir (`hidden_alpha`), Blender'daki "hidden line"
  görünümü gibi.
- Sonuçlar önbellekte tutulur. Önbellek kamera, geometri revizyonu veya timeline
  karesi değişince boşalır.
- Kare başına 6.000 ışın bütçesi var; aşılırsa kalan çizgiler eski görünümle,
  önde çizilir.
- Taşıma/döndürme tutamaçları (ImGuizmo) bilerek hep önde kalır.

Bağlanan gizmolar:
- sim domain kutuları, seed kutuları;
- collider'lar: OBB, AABB, düzlem, küre, kapsül, SDF;
- flow-source küreleri;
- kuvvet alanının yön oku ve sınır şekli (küre, kutu, silindir, koni).
Kuvvet alanının ortadaki simgesi tıklanabilir olduğu için önde kalır.

Ayar: `viewport_settings.gizmo_occlusion` (varsayılan açık) ve
`gizmo_hidden_alpha` (0,25). Panelde gizmo düğmesine sağ tık →
"Gizmo Depth" ve "Hidden Alpha". Proje kaydına yazılıyor. IPC'de
`viewport.set_gizmo_occlusion` ve `viewport.get_gizmo_occlusion`; Python'da
`rt.viewport.*`. Yeni dosya `.vcxproj` ve `.filters`'a eklendi. Shader değişikliği yok.

Sıra: önce bağımsız ve hızlı olanlar.

1. **Derleme.** Hata çıkarsa ilk bakılacak yerler:
   - `GizmoOcclusion.cpp` içindeki `extern g_scene_geometry_generation` (tanımı
     `Core/Main.cpp:639`'da).
   - `HitRecord` (`Hittable.h`).
   - `scene_ui.cpp` ve `RtApiViewport.cpp`'ye eklenen include'lar.
2. **Gözle kontrol:** bir domain kutusunu bir mesh'in arkasına koy, perspektifte bak.
   - Mesh'in arkasında kalan kenarlar soluk, önündekiler tam renkli olmalı.
   - Sağ tık → "Gizmo Depth" kapatılınca eski görünüm (her şey önde) gelmeli.
   - ★ **Sinsi durum:** bir collider kutusu kendi nesnesine yapışık; kenarları
     yüzeyin üstünde ve tamamı soluk çiziliyor. Bu, toleransın yetmediğini
     gösterir (`kRelativeTolerance`). "Kutu nesnenin içinde" gibi makul
     göründüğü için kolay gözden kaçar.
3. **Maliyet (IPC):** `viewport.get_gizmo_occlusion`.
   - Kamera dururken `rays` ≈ 0 olmalı (önbellek çalışıyor).
   - Kamerayı döndürürken `rays` birkaç yüz ile birkaç bin arasında olmalı.
   - `budget_exhausted=true` ise bütçe dolmuş ve kalan çizgiler önde çiziliyor
     demektir; bunu söyle, bütçeyi ölçerek ayarlayalım.
   - `active=false` ise `inactive_reason` kapının adını verir.
4. **Rendered modu ve ortografik:** üst görünümde (ortografik) ve Rendered
   modunda da aynı davranış beklenir. Ortografikte ışınlar paraleldir.

Henüz bağlanmayanlar (sonraki tur): flow-source yön okları, ışık ve kamera
gizmoları, emitter yönü, rig/terrain overlay'leri.
