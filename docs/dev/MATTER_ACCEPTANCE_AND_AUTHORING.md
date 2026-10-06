# Matter — kalan kabul ve sonraki veri/UI sırası

2026-10-05. C5/C6 kapalı bilanço ve temel GPU probları geçti. Tam fiziksel
kapanış henüz yok. UI/veri sahipliği göçü fizik kabulünden sonra gelir.

## H1 üretim entegrasyonu — hedef teyidi (2026-10-05)

Kanonik yön: [Ana plan / H1 üretim hedefi](BIRLESIK_MADDE_DOMAIN_TASARIMI.md#h1-üretim-hedefi--ortak-gpu-madde-tek-domain-2026-10-05).
Tek Matter domain, ortak GPU state ve model başına gereken ayrı scratch/grid;
aynı taşıyıcının tek transport owner'ı. Yüksek yoğunluk için DEM ve PBD/XPBD
adayları ölçülür; CPU DEM küçük referans olarak kalır. Önce emitter/flat collider/
GPU render ile domain içinde granül runtime; otomatik MPM↔grain geçiş daha sonra.
Yerel water/temperature/bond/damage ve korunumlu dönüşümler aynı state'i kullanır.
Normal çözücü geçişinde full-state CPU download/upload yok; sorgu/cache ayrı ölçülür.

H1-R kısa referans 21/21 PASS; H1-G0/G1/G2, H1-P/H, GPU resident C7 açık.
Yeni authoring işlemlerinin core/UI/Python/IPC/validation/persistence parity'si
ilgili paketten ayrı ertelenmez. Aşağıdaki genel editable profile UI göçü ayrı iştir.

## Önce kapanacak kapılar

| Kapı | Gereken kanıt | Durum |
|---|---|---|
| C5 korunum/drenaj | Kaynak durduktan sonra dry/free/pore kg; doygunluk sınırı; havuz baskısı | İlk kısa koşu PASS; uzun havuz baskısı açık |
| G2/C6 mekanik | İki yığın yüksekliği, kuru/nemli/doygun çökme/yayılma/repose | Açık; endpoint constitutive test kaynakları mevcut |
| C6 mekân/görünüm | Canonical saturation bandlarının konumu; kuru kontrolün gerçekten kuru kalması; renderer kontrolü | İlk wet palette oluştu; uzun koşuda sağ kontrol ıslanıyor |
| C6 kalıcılık | Aynı yakalanmış frame'de kimlik, kg, saturation bands, konum; save/load ve cache/scrub/resume | Yeni read-only snapshot karşılaştırma aracı hazır; build/canlı test bekler |
| dt/grid | Aynı fiziksel boyut/yük, 1/60 ve 1/120; h=0.1 ve 0.08 | Araç seçenekleri hazır; sonuç yok |
| C4/C7 | Tek su/tek kum/karma A/B; authored stiffness/substeps; %10 kapılar; ledger/render/preset regresyonu | Açık; karma wet açık/kapalı koşuları tek-faz performans kanıtı değildir |

Ampirik u=rho*g*h*S², çözünürlükten bağımsız pressure PDE değildir.
Tam hidrostatik pressure/drag/buoyancy kanıtı ve algoritması ayrı açık kalır.
Bu sınırlar ölçülmeden plan TAMAMLANDI olmaz.

## Sonraki kullanıcı derlemesi ve dış test

Yeni MatterAcceptanceMetrics.cpp/.h proje listesine eklendi; shader değişmedi.
`fluid.matter_models` ve mevcut Python karşılığı aynı raporu döndürür:
granular dry/pore kg, capacity-weighted saturation, dry COM, material-point
bounds, horizontal RMS radius, transport kinetic energy ve 8 spatial band.
Boş bounds/COM null; bunlar görünür sphere radius veya repose ölçümü değildir.
Güncel görünüm bandları A=clamp(S/wet_appearance_full_saturation,0,1), ceil(7*A)
ile seçilir; default threshold .05. Mean saturation hâlâ fiziksel pore/capacity.
Restore probu normalized_saturation_upper_edge_v2 policy ve threshold eşitliğini
de denetler; önceki snapshotlar yeni policy için yeniden alınır.
Legacy eksik/unwritten mass sidecar'ında mevcut genel rapor korunur; yeni
acceptance_metrics measured=false/reason döndürür, tahmini kg kabul kanıtı olmaz.
Yeni C++ regression: scripts/test/matter_acceptance_metrics_test.cpp.

Duraklatılmış kabul sahnesi için dış terminal:
```
python scripts/test/rt_c5_acceptance_scene_ipc.py --setup --wet --steps 25 --log docs/dev/c6_dt60.json
python scripts/test/rt_c5_acceptance_scene_ipc.py --steps 180 --balance --log docs/dev/c6_dt60.json
python scripts/test/rt_c5_acceptance_scene_ipc.py --setup --wet --dt 0.008333333333 --steps 50 --log docs/dev/c6_dt120.json
python scripts/test/rt_c5_acceptance_scene_ipc.py --dt 0.008333333333 --steps 360 --balance --log docs/dev/c6_dt120.json
python scripts/test/rt_c5_acceptance_scene_ipc.py --setup --wet --voxel 0.08 --steps 25 --log docs/dev/c6_h008.json
python scripts/test/rt_c5_acceptance_scene_ipc.py --steps 180 --balance --log docs/dev/c6_h008.json
```
Setup açık sahneyi değiştirir/reset eder. Voxel koşusunda kaynak sayısı/oranı
voxel hacmine ters ölçeklenir; integer rounding nedeniyle gerçek kg ayrıca
raporlanır, eşit başlangıç kütlesi varsayılmaz. Zaman adımı/voxel testleri henüz
çalışmadı. `--water-limit 0` ayrı kuru referans kurar; doygun referans yerine geçmez.
`--balance` fizik bilançosunu denetler; aktif kaynak veya timeline değişimi
koşuyu reddeder. Şekil karşılaştırmaları fiziksel kabul olmadan PASS ilan edilmez.

Yakalanmış, durmuş aynı timeline frame için:
```
python scripts/test/rt_test_matter_snapshot_ipc.py C5_WaterSand_Acceptance --write docs/dev/c6_snapshot.json
# Kullanıcı kaydet/yükle veya bake/scrub ile aynı yakalanmış frame'i geri getirir.
python scripts/test/rt_test_matter_snapshot_ipc.py C5_WaterSand_Acceptance --compare docs/dev/c6_snapshot.json
```
Probu cache olmayan manuel fluid.step frame 0 durumundan timeline frame 0'a
karşılaştırma: timeline o runtime'ı temsil etmez. Görsel render ayrıca kontrol edilir.

## Sonra veri sahipliği ve UI

| Sahip | Kanonik sorumluluk | UI anlamı |
|---|---|---|
| Matter domain | Bounds/grid, sınır, solver/bütçe, ortak exchange/contact | Water seçmek domain'in tamamını Water'a dönüştürmez |
| Düzenlenebilir madde tanımı | Kararlı kimlik; yoğunluk, viskozite; granular friction/cohesion/Young/dilatancy; thermal/phase özellikleri | Preset başlangıç değerleri sağlar; özel madde oluşturulabilir |
| Emitter | Madde tanımına referans, miktar/zaman/hız/başlangıç sıcaklığı | Granular bu emitter'ın seçtiği maddenin constitutive modeli; resolved özellikler görünür |
| Scene görünüm materyali | BSDF/texture; fizik tanımıyla açık ilişki | Fizik profilinden ayrı render görünümü; wet varyantlar türetilir |
| Matter Output | Taşınan maddelerin görünüm/representation eşlemesi ve kullanım bilgisi | Yeni madde üretmez; emitter/live parcel/binding kaynakları açıkça ayrılır |

Bugünkü domain-global granular params, string substance/preset lookup ve bağımsız
output binding bu hedefin tam uygulaması değildir. Aynı domain'de iki farklı
granüler maddenin parametrelerini ayrı uygulamak GPU constitutive veri göçünü
de gerektirir; yalnız widget taşıyarak çözülmez.

Sıra: ortak editable profile core + stable references → parcel/GPU çözümleme →
serializer/cache migration + rename/delete/reference politikası → ortak
Python/IPC işlemleri/transactional validation → mevcut context rail/dock içinde UI.
Emitter override gerekiyorsa açık bir profile instance oluşturur; paylaşılan
tanımı sessizce değiştirmez. Aynı maddeyi taşıyan iki emitter ortak özellikleri
görür. Silinen bir preset/material, live parcel kimliğini sessizce yeniden yazmaz.
Kullanımdaki silme/yeniden eşleme ve sahipsiz binding temizliği core politikası olur.

Bu aşama henüz uygulanmadı; kullanıcı fizik planından sonraya istedi.
