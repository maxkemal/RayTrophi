# 2026-10-11 parti 4: sanal sıcaklık, yoğuşmayan gaz havuzu, rüzgârla yağmur

Parti 3 (yağmur) canlı 9/9 PASS. Bu parti madde planının açık listesindeki 3, 4
ve 5. maddeleri kapatıyor. Ayrıntı: [BIRLESIK §4.8c](BIRLESIK_MADDE_DOMAIN_TASARIMI.md).

- **3. Sanal sıcaklık.** Gaz sıcaklık alanı Tv tutuyor; kaynayan buhar 600 K
  okunuyordu. Doygunluk ayarı ve yüzey alışverişi artık hücrenin buhar kütlesiyle
  gerçek hava sıcaklığına çeviriyor ve değişikliği alana Tv FARKI olarak yazıyor.
- **4. Yoğuşmayan gaz.** Envanterde 12. yuva (`gas_phase_foreign_kg`). Yanmanın
  yakıt buharı ve yoğuşabilir madde dışında kaynayan her şey buraya gider. Yüzey
  buharlaşması yalnızca yoğuşabilir maddeye uygulanır.
- **5. Rüzgâr.** Yağmur kütlesi, enerjisi ve yoğuşmayan gaz, gaz hızıyla
  semi-Lagrangian taşınıyor. Düşüş önceki gibi ayrı.

Yeni shader yok. Değişen shader'lar: `sim_inventory_saturation.comp` (bağlama ve
itme sabiti aynı) ve `sim_inventory_surface_exchange.comp` (itme sabiti
**36 → 40 B**). Derleme: `compile_sim_shaders.bat`. Disk önbelleği **v15**
(v11–14 okunur, yabancı havuz boş gelir).

Sıra: önce bağımsız ve hızlı olanlar, sonra diğerlerini maskeleyebilecekler.

1. **Derleme.** Hata çıkarsa ilk bakılacak yerler:
   - `GasInventoryDevice::fields` 12 elemanlı oldu.
   - `persistentImages` 6 eleman döndürüyor.
   - `GasInventoryHostImage` içinde yeni `foreign` alanı var.
   - `GasInventorySummary` içinde `foreign_kg`, `foreign_cells`, `rain_centroid`
     alanları var.
   - Yeni fonksiyonlar `virtualKelvinFromAir` ve `airKelvinFromVirtual`
     (`GasSaturation.h`) ile `depositGasForeign` (`GasInventoryGpu.h`).
2. **Kaynak sözleşmesi (derleme gerektirmez):**
   `python scripts/test/check_gas_inventory_gpu_contracts.py` → PASS. Bu partide
   PASS verdi.
3. **.spv zaman damgası.** `sim_inventory_surface_exchange.spv` eskiyse itme
   sabiti uyuşmaz ve `inventory_step_held` "gas surface …" hatası verir. Önce
   shader derlemesini doğrula.
4. **Regresyon, sis ve yağmur:** `python scripts/test/rt_test_fog_ipc.py`. Önce
   rüzgârsız koş; bu koşu taban dosyasını (`fog_live.json`) yazar. 10 hüküm var;
   yeni olan `water_stays_vapour`.
   - `water_stays_vapour` FAIL: su kaynaması yabancı havuza gidiyor. Bu
     durumda `resolveCondensableVapour` ile parçacık maddesinin adı eşleşmiyor
     demektir; Default Substance etiketsiz mi, ona bak.
   - `water_conserved` FAIL ve fark yaklaşık `gas_foreign_kg` kadar: aynı sebep.
   - ★ **Sinsi durum:** tüm hükümler PASS ama `fog_condensed_kg` önceki koşudan
     (`docs/dev/fog_live.json`; otokonversiyon 94 kg) belirgin biçimde DÜŞÜK.
     Beklenen yön ARTMASI. Kaynağın üstündeki buhar artık 600 K değil gerçek
     sıcaklıkla okunuyor, yani daha çok yoğuşmalı. Düşükse Tv dönüşümü ters
     çalışıyor ya da `pc.molar` sıfır geliyor. Bunu kimse hata diye raporlamaz,
     o yüzden sayıyı elle karşılaştır.
5. **Rüzgâr:** `python scripts/test/rt_test_fog_ipc.py --wind 4`. Taban dosyası
   olarak 4. adımın çıktısını okur. Hüküm `rain_drifts_downwind`: yağmurun ağırlık
   merkezi tabandan en az bir voksel (0,05 m) +x yönüne kaymış olmalı.
   - FAIL ve fark ≈ 0: önce `gas_rain_centroid`'in sıfırdan farklı geldiğine
     bak (özet indirgemesi). Sıfır değilse rüzgâr alanı gaza ulaşmıyor
     (`affects_gas`). Bu durumda sorun yağmurda değil.
   - FAIL ve fark negatif: kapalı kutuda dönüş akımı yağmuru geri taşıyor.
     Ölçüm doğru, test geometrisi yanlış. Açık sınırla tekrar dene; kodu
     değiştirme.
6. **Yoğuşmayan gaz:** `python scripts/test/rt_test_gas_species_ipc.py`. Boş ve
   duraklatılmış bir sahnede koş: `burning_fuel_spill` preset'ini ekler, sonra
   siler. Dört hüküm var: `combustion_ran`, `fuel_in_foreign_pool`,
   `fuel_not_vapour`, `no_fuel_fog`.
   - `combustion_ran` FAIL: preset tutuşmadı. Diğer üç hüküm o zaman anlamsızdır.
7. **GPU envanter regresyonu:** `python scripts/test/rt_test_gas_inventory_gpu_ipc.py`.
   Kaynama ve ısı alışverişi fixture'larını kullanır. Havadaki kütleye artık
   `gas_foreign_kg` de dahil.
   - `inventory_grid_upload/download_bytes` 0 değilse: yeni yuvaların taşınımı
     host'a gidip geliyor demektir. Bu olmamalı.
8. **Ölçülmedi, bu partide kapanmadı:** yağmur parsellerinin DOĞUM YERİ (açık
   listede madde 2). Bunun için parsel konumu okuyan bir ölçüm gerekiyor; bu
   partide yazılmadı.

## Bu partinin bilinen sınırları (hata değil, karar)

- Yanıcı olmayan başka bir sıvı (yoğuşabilir madde dışında kalan) yüzeyden artık
  hiç buharlaşmıyor. Önceden su buharı havuzuna buharlaşıyordu, bu yanlıştı.
  Kendi tür havuzu yok.
- Kaynayan yabancı maddenin Tv izi alanda kalıyor. Dönüşüm yalnızca su buharı
  kütlesini sayıyor. Ağır yakıt buharı biraz "soğuk" görünebilir.
- Yanıcı sıvının mist damlası hâlâ sis suyu havuzuna gidiyor (sıvı → sıvı, faz
  değişimi yok). Ayrı bir sıvı türü havuzu yok; açık.
