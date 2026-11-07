# DEM active work queue — revision 38

Durum: kullanıcı build'i ve ilk canlı çalıştırma doğrulandı; geniş fizik/performance kabulü açık.
İlk canlı scheduler/performance ölçümü tamamlandı; sonuç ve kabul sınırları:
[DEM_ACTIVE_WORK_PROFILE_2026_11_06.md](DEM_ACTIVE_WORK_PROFILE_2026_11_06.md).
Önceki ölçüm: [DEM_PERFORMANCE_PROFILE_2026_11_06.md](DEM_PERFORMANCE_PROFILE_2026_11_06.md).

## Davranış

`compact_sleep_work` varsayılan true. Önceki başarılı adımın uyuyan oranı en az
%25 ise, sleep etkinse, history/context değişmemişse kompakt yol seçilir.
Diğer durumlarda mevcut dense yol kullanılır. Hareketli collider, etkin MPM
contact veya ortak continuum saati zaten sleep'i kapattığından kompakt yol da
kullanılmaz. Support removal/context wake ve fresh history bir dense adım alır.

Kompakt alt adım sırası:

1. `list_clear`: rebuild flag rotasyonu ve active count sıfırlama.
2. `hash`, `list_build`: mevcut Verlet listesi davranışı, bütün taneler için.
3. `active_classify`: hafif tane başına sleep seçimi. Gerçekten atlanabilir
   sleeping taneler karşı bankaya taşınır, skipped spring time gerçek dt kadar
   ilerletilir. Diğer tanelerin kimlik indeksleri GPU listesine eklenir.
4. `step`: yalnız listedeki tane indeksleri heavy contact/history/BVH yoluna
   girer. Boş indeks aralığı history/state okumalarından önce döner.

Read bank, bütün pair okumaları boyunca önceki alt adımın tamamıdır. Classifier
yalnız write bank ve sleeping tanenin kendi elapsed word'ünü yazar. Aktif liste
aynı Scratch SSBO'nun state'ten sonraki alanındadır: ilk `9*N` float state,
sonraki `N` word bitcast edilmiş tane indeksleri. Tahsis/VRAM bütçesi `10*capacity`
float olarak güncellendi; descriptor 17 ve push constants 128 byte kaldı.

## Korunan wake/audit kuralları

- FRESH veya kimliği doğrulanmamış history asla skip edilmez.
- Hız/spin, coupling ve context kuralları aynı sleep politikasını kullanır.
- Periyodik audit ve final substep bütün gerekli sleeping taneleri heavy yola alır;
  final contact/CFL sayaçları yalnız aktif tanelerle sınırlandırılmaz.
- Classifier AUDIT bitini temizlemez. Pending request heavy yola gider.
  Classifier'dan sonra aktif çözücü tarafından gelen istek, sonraki alt adımda
  okunur. Eski dense yol da thread sıralamasına göre isteği aynı veya sonraki
  alt adımda görebiliyordu; bu nedenle karşılaştırma bit-identical trajectory
  iddiası değildir. Impulse/wake, değişken kütle ve repose kabulü gereklidir.
- Contact stiffness, damping, friction, rolling, sleep eşikleri ve rest süresi
  değiştirilmedi. Geçersiz durum/overflow reddi ve host publication korundu.

## API ve UI

İki binding aynı `rtapi::matterGrainSettings` -> `patchMatterGrainParams` core'unu
kullanır:

```python
rt.fluid.set_grain_settings("Physics Domain 1", compact_sleep_work=True)
# Dense scheduler reference; sleep itself remains enabled:
rt.fluid.set_grain_settings("Physics Domain 1", compact_sleep_work=False)
```

IPC: `fluid.set_grain_settings` için `domain` + `compact_sleep_work` JSON boolean.
Integer/string/unknown key transactional olarak reddedilir. Ayar kaydedilir ve
`fluid.grain_settings` ile okunur. Yalnız scheduling/probe değişimi cache temizlemez;
fizik parametresi de değişmişse mevcut invalidation sürer. UI Advanced solver'daki
`Compact sleeping work` aynı core'a gider.

Gerçekte seçilen yol `fluid.matter_models(domain=...).grain_diagnostics.runtime`
içindeki `sleep_work_mode` (`dense`/`compact`). True ayar her karede compact demek
değildir; önceki oran ve güvenlik kapıları uygulanır.

## Performans sınırı

Compute API henüz indirect dispatch sağlamıyor. Heavy kernel yine N için dispatch
edilir, ama aktif indeksleri bitiren invocation'lar girişte döner. Kazanç adayı,
uyanık tanelerin ardışık GPU iş aralığına toplanması ve sleeping carry'nin küçük
kernel'de çalışmasıdır. Dispatch sayısı compact modda **5*substeps**, dense modda
**4*substeps**; dispatch sayısının azaldığı iddia edilmez. Classifier/atomik append
maliyeti nedeniyle az uyuyan yığında kazanç garanti değildir; %25 adaptif kapı bu
yüzden vardır. Profil karşılaştırmasına classifier zamanı da dahil edilmelidir.

## Doğrulama ve kullanıcı build listesi

Ajan build, shader compilation veya uygulama açılışı çalıştırmadı.

1. C++ ve **beş DEM shader'ını** birlikte derle: list_clear, hash, list_build,
   **active_classify**, step. Mevcut `compile_sim_shaders.bat` yeni kernel'i içerir.
   Revision **38**, diagnostics/audio header **47 word**; runtime'ın okuduğu
   shader dizininde hepsi güncel olmalı.
2. `matter_grain_stages_test.cpp`: dense/compact stage sırası, ping-pong son bankası,
   her aşamada hata yayılımı. `matter_grain_params_test.cpp`: boolean validation ve
   compact seçeneğinin diğer sleep değerlerini koruması. Bu C++ testleri derlenmedi.
3. Build sonrası **ayrı boş test sahnesinde**, mevcut dış IPC testleri:
   `rt_test_grain_sleep_ipc.py` (unsupported free fall, supported sleep,
   compact/dense aynı floor yüksekliği), `rt_test_grain_sleep_transfer_ipc.py`
   (komşu momentum/uyanma), ayrıca wet/motion/repose kabulü.
   Kullanıcının açık üretim sahnesinde bu fixture'lar çalıştırılmadı.
4. Gerçek sahnede yüksek uyku oranında `sleep_work_mode=compact` doğrula.
   Probe açıkken solves+skips=grain_count*substeps ve final snapshot bütünlüğünü
   `probe_dem_collider_ipc.py` ile kontrol et. Probe kapalıyken GPU timestamp ile
   dense/compact karşılaştır: step + classifier + list kernels ve host toplamı.
   Aynı kaydedilmiş başlangıçtan iki kol en güçlü fizik/performance kıyasıdır;
   ardışık farklı durumlar kesin A/B değildir. Mass, destek, hareket sonrası
   uyanma ve enerji/şekil toleransları süre kadar önemlidir.

Non-build kontrolleri: `check_matter_grain_contracts.py`,
`check_matter_grain_active_work.py`, `check_matter_grain_sleep_contracts.py`,
`check_matter_grain_sleep_transfer.py`. Bunlar GPU hızlanması/fizik paritesi kanıtı değildir.
