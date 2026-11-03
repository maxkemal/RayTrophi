# Mesh UV araçlarının bağımsız editöre taşınması

Tarih: 2026-11-04. Durum: kaynak ve statik kontroller tamam; yeni build/canlı kabul bekleniyor.

Materyal panelindeki UV Workflow kaldırıldı. 300 üçgenlik önizleme, facade cache'i,
UI içindeki normalize/flip/swap lambdaları ve kullanılmayan facade view-projection
fonksiyonu silindi. SceneUI'nin büyük header'ına yeni alan/metot eklenmedi.

## Sorumluluk ayrımı

- UV Editor: tam UV canvas, ada seçimi, transform/pivot, uniform normalize, Flip U,
  Flip V, Swap U/V, Project from View, auto unwrap ve pack.
- Material > Texture Coordinates: seçili UV seti, texture scale/offset/rotation,
  axis lock, Wrap, Reset UV Transform ve seçili nesneyi açan Open UV Editor.
- Materyal örnekleme ayarları `UI/MaterialUvSamplingUI` modülüne çıkarıldı.
  Çok materyalli flat sibling'lerde ortak UV setleri gösterilir. Önceden var olan
  aktif setin render `uv` kanalına kopyalanma davranışı korunur; set-zero/render
  kanalı ayrımı bu taşımanın tamamlandığı anlamına gelmez ve roadmap'te açıktır.
- UV editörü yayımlanmamış inspector helper header'larına bağımlı değildir;
  mevcut public UIWidgets API'sini kullanır. Diğer ajanın materyal/domain işi
  UV commit'ine dahil edilmez.

## Ortak operasyonlar

Yeni taşınan işlemler UI'da UV::request üzerinden, Python'da keyword argümanlarıyla
`rt.uv.flip_u`, `flip_v`, `swap_uv`, `project_view`; IPC'de `uv.*` olarak sunulur.
13 IPC/Python yöntemi ve discovery şemaları adaptörde kayıtlıdır.

Flip/Swap seçilen adalara uygulanır; boş seçim bütün adaları hedefler.
Flip ekseni UV tile merkezi 0.5'tir. Swap sayısal koordinatları doğrudan değiştirir.
Project from View bütün seçili mesh'i aktif sahne kamerasından projekte eder;
perspektif/orthographic kamera ayrımı ve world-space P kanalı kullanılır.
Sonuç 0–1 alanına uniform ölçeklenir. Kamera eksenleri geçersizse, kullanılan bir
vertex kameranın arkasında/near plane'de ise veya projeksiyon dejenere ise işlem
UV'leri değiştirmeden açık hata verir. Bunlar kamera viewport'undan kaynaklanan
overlap'ları açan bir unwrap solver değildir.

Tüm mesh yazmaları mevcut revision/render-lock/texture-change/shared-geometry
kontrollerinden ve tek SceneHistory transaction'ından geçer. Malzeme sampling
dönüşümleri mesh UV'lerine normalize/flip işlemi uygulamaz.

```python
info = rt.uv.info(object="Default_Cube", mesh=0)
rt.uv.flip_u(object="Default_Cube", mesh=0, revision=info["revision"],
             islands=[0], allow_texture_change=True)
info = rt.uv.info(object="Default_Cube", mesh=0)
rt.uv.project_view(object="Default_Cube", mesh=0, revision=info["revision"],
                   allow_texture_change=True)
```

## Doğrulama

- `python -B scripts/test/check_uv_authoring_contracts.py`: altı statik kontrol geçti.
- Kabul probe'u Flip U/V/Swap için seçim izolasyonu, koordinat sonucu ve undo
  kontrolleriyle genişletildi. Projection finite/sınır/topoloji ve atomik ret kontrolü eklendi.
- Bu turda build veya uygulama açılışı yapılmadı. Önceki build üzerinde geçen IPC,
  Python, normalize/undo ve xatlas testleri [canlı test kaydında](UV_EDITOR_LIVE_ACCEPTANCE_2026_11_03.md).
- Yeni build sonrası: materyalde eski UV Workflow/preview bulunmamalı; sampling
  Scale/Offset/Rotation/Wrap değişmeli ve resetlenmeli; editör kısayolu doğru nesneyi
  açmalı. UV Editor'da taşınan araçlar ve Ctrl+Z/Ctrl+Y kontrol edilmeli.
- Ardından ayrı terminalde `python scripts/test/rt_test_uv_authoring_ipc.py` çalıştırılmalı.
  Probe geçici test küpünü siler ve önceki UV hedefini tekrar açar; proje/history modified olur.

Tam UV authoring hedefi (corner remap, seam/pin, conformal solve, texture rebake ve
GPU canvas) [ana yol haritasında](UV_AUTHORING_AUDIT_ROADMAP.md) devam eder.
