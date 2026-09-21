# Sıradaki derlemede kontrol edilecekler

> **Durum:** CANLI — 2026-09-21 (ikinci parti). Sıra: bağımsız ve hızlı görülen önce.

★ **Hem C++ hem SHADER değişti.** `compile_shaders.bat` çalıştırılmadan
`sim_gas_buoyancy.comp` değişikliği etkisizdir ve **hiçbir belirti vermez** —
eski .spv sessizce yüklenir ve CPU yolu ile GPU yolu ayrı fizik koşturur.

---

## 0. ⚠ BEKLENEN DAVRANIŞ DEĞİŞİKLİĞİ — bunlar regresyon DEĞİL

1. **Soğumuş bulut artık daha HIZLI iniyor**, daha yavaş değil. Clamp
   kaldırıldı; bkz. madde 2.
2. **Bütün ateş/duman sahneleri soğuk bölgelerde çok daha karanlık** (T⁴).
3. `buoyancy_heat` ve `ambient_stratification` **yeniden ayarlanmalı.** Eski
   sayılar artık var olmayan bir sınıra göre seçilmişti.

---

## 1. Derleme geçmeli — sessiz olan şey artık gürültülü

`buoyantAnomaly()` bir parametre kazandı (`density`). İki çağrı yeri
güncellendi (yoğun CPU, seyrek VDB). Üçüncüsünü kaçırdıysan **derleyici
söyler** — amaç buydu.

`static_assert(sizeof(GasBuoyancyGpuConstants) == 44)` de yerinde.

## 2. ★★★★★ Kaldirma çöküşü BITTI — doğrula

`ambient_stratification = 0.25` ile ölçüldü (surface dust açık ve kapalı,
ikisinde de aynı): merkez 19.4 → 22.6 → 23.6 → **23.6**, f160-f200 arası fark
**0.04 m**. Eski hâli: 18.5 → 17.1 → 13.6 → 8.5.

**Ne görmen gerek:** `gas.measure_plume` → `centroid_above_floor` f160 ve
f200'de aynı.

★★ **ÖLÇÜ ALETİ UYARISI — iki metrik de YANILTIR:**
- `top_above_floor` f130'dan sonra **tavanda doyar** (34.0) ve kapağın
  hareketini gösteremez. Her yoğunluk eşiğinde 34.0 döndü.
- `centroid_above_floor` zemin eteğinin kütlesiyle domine.
- Gözle görülen "çökme" bulutun **ALT cephesinin** inmesidir ve bunu ikisi de
  ölçmez. Ekran görüntüsü al, sayıya güvenme.

## 3. ★★★★★ PRESET'TEN YENİ SAHNE: duman incelerek yükselmeli

Üç preset değeri düzeltildi (`SceneDataParticlePresets.cpp`, nükleer kolu):

| alan | eski | yeni |
|---|---|---|
| `gas_density_dissipation` | 0.012 | **0.18** |
| `gas_ambient_stratification` | `6.5f/(26.0f*S)` = 0.25/S | **0.05f / S** |
| `emission.temperature_max` | 5000 | **20000** |

**Ne görmen gerek:** koyu gri sütun, yükseldikçe incelen yarı saydam duman,
közlü kapak, dağılan zemin katmanı — ve önemlisi, `gas.measure_plume` →
`active_cells` f110 civarında **tepe yapıp geri düşmeli** (1.29M → 1.19M),
monoton artmamalı. `fill_fraction` ~0.35'te sabitlenmeli.

**Bozuksa ne demek:**
- `active_cells` monoton artıyor ve fill 0.6+ → `density_dissipation` uygulanmamış.
  `gas.get_settings` ile oku; 0.012 görüyorsan `gas_dissipation_override`
  kapanmış olabilir.
- Kapak düz beyaz → `temperature_max` eski değerde.

★ **En sinsi hâli:** bulut "makul" görünür ama hiç incelmez. Bu gözle
yakalanmaz çünkü tek kare güzeldir; f110 ve f200'de `active_cells`
karşılaştır, tek teşhis budur.

★ Kaynaklar zaten 4.5 s'de (f108) bitiyor — `flow_source.list` ile doğrulanabilir.
f108 sonrası kütle artıyorsa bu **yayılmadır**, üretim değil.

⚠ Kalan kusur: kapak f110'dan sonra hâlâ domain tavanında (34·S m).
Parametreyle çözülmez; domain yüksekliği ayrı bir karar.

## 4. ★★★ Sıralama testi: soğuk gaz ılık gazdan DAHA HIZLI inmeli

Eski davranış tersineydi: ılık gaz batıyor, tamamen soğumuş gaz asılı kalıyordu.

**Ne görmen gerek:** bulut soğudukça iniş **hızlanır**. Kapak artık "sıcakken
düşüp soğuyunca duran" bir şey değil.

**Bozuksa ne demek:**
- Hiçbir şey değişmediyse → shader derlenmemiş VEYA CPU yolu koşuyor. Hangi
  yolun koştuğunu `gas.step_stats` söyler.
- Bulut anında yere çakılıyorsa → `ambient_stratification` artık çok yüksek.
  0.25 clamp'li terime göre seçilmişti; **0.05–0.10 aralığından başla.**

## 5. ★ Boş alan hâlâ kuvvet almamalı

Bu, clamp'in var olma sebebiydi ve artık yapı gereği sağlanması gerekiyor.

**Ne görmen gerek:** dumansız bölgede hiçbir hareket yok; domain genelinde
aşağı doğru bir akım yok.

★ **En sinsi hâli:** domain çapında zayıf bir aşağı akım — hiçbir şeyi
bozmaz, sadece bulutu "makul biçimde" oturtur ve kimse bunu bug diye
raporlamaz. `gas.measure_plume` ile boş sahnede `top_above_floor`
düşüyor mu diye bak.

**Bozuksa:** `presence` kapısı yoğunluk kanalını görmüyordur
(`pc.has_density`).

## 6. Yoğunluk kanalı KAPALI bir domain — tabakalaşma KAPANMALI

Kasıtlı: dumanı havadan ayıramıyorsak tahmin etmiyoruz.

**Bozuksa:** sessizce eski hayali aşağı akım geri gelmiştir.

## 7. ★★★★ Zayıf bölgelerin BEYAZLAMASI bitmeli

`temp > 20` bir **birim tahmini**ydi ve tam da neredeyse boş hücrelerde yanlış
cevap veriyordu: solver ısısı 0.005 olan hücre x3000 sonrası 15 oluyor, testi
geçemiyor, normalize kola düşüyor ve `clamp(15,0,1)` 1.0'a doyup **rangeMAX**
döndürüyordu. Domain'in en sönük gazı en sıcak olarak raporlanıyordu.

T⁴ bunu görünür yaptı: o hücreler radyans 1.0'da otururken 10000 K'lik gerçek
sıcak gaz 0.0625 alıyor — **en soğuk gaz ateş topundan 16 kat parlak.**
Yoğunluk cutoff'unu yükseltmek gizliyordu, çünkü o hücreleri tamamen atlıyor.

**Ne görmen gerek:** cutoff'u DÜŞÜR (0.007 veya altı) — zayıf bölgeler artık
beyaz benekler üretmemeli, sönük ve gri kalmalı.

**Bozuksa:** shader derlenmemiş.

## 8. Sıcaklık–renk: kapak ton kazanmalı

`temperature_max` artık **iki iş birden** yapıyor: rengin doyduğu nokta ve
**radyansın normalize edildiği nokta.**

Ölçülen: sahnenin kelvin aralığı 5475–19769, `temperature_max` 5000 iken
**kapağın tamamı radyans 1.0'a kıstırılıydı** — beyaz levhanın sebebi buydu.
Sahnede 20000'e çekildi ve kapak anında ton kazandı.

**Bozuksa:**
- Kapak hâlâ düz beyaz → `temperature_max` sahnenin gerçek aralığının altında.
  `gas.measure_plume` → `peak_temperature × 3000` ile karşılaştır.
- Her şey karardı → aynı değer çok yukarıda.

## 9. Renk rampalı sahneler DEĞİŞMEMELİ

T⁴ yalnızca blackbody koluna uygulandı. Rampalı hacim birebir aynı olmalı.

## 10. Üç render yolu birbirini tutmalı

Raster viewport, Vulkan RT, CPU/OptiX. Dördüncü çağrı yeri: `volume_closesthit.rchit`
içindeki emitter bloğu (~3781), ışık yayan hacmin komşu yüzeyleri aydınlatması.

---

## Kod dışı kalanlar (bilinçli)

- **`temperature_scale` IPC'ye açılmadı.** Tonu radyanstan bağımsız kaydıran
  tek dial ve T⁴'ten sonra asıl kalibrasyon aleti bu. Kural 1'e göre açılmalı.
- **Sahnedeki değerler preset'e girmedi.** `SceneDataParticlePresets.cpp` hâlâ
  `ambient_stratification = 0.25` ve `temperature_max = 5000` taşıyor. Madde
  0.3 yüzünden senkron ancak yeniden ayardan SONRA anlamlı.
- **Domain tavanı.** f110'dan sonra `touching_ceiling`. Domain 22.1 × 34 × 22.1.
