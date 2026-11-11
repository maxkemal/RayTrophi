# Yerleşik preset düzeltmeleri — 2026-11-08

Bu değişiklik gerçek fizik davranışını bazı yerlerde değiştirir; yeni bir
non-Newtonyen çözücü eklemez. Isıl/kimyasal profiller ve 26 somut örnek korunur.
Önceki 2026-11-07 envanteri tarihsel durumdur; bu belge yeni durumu açıklar.

## Yapılan düzeltmeler

| Alan | Önce | Sonra / gerekçe |
|---|---|---|
| Snow, Ice eriyik ν | 0 | Water profilinin sıvı ν ve yoğunluğu; katı granular yol zaten sıvı viskozitesini devre dışı bırakıyor |
| Honey, Oil, Chocolate body drag | 0.2 / 0.15 / 0.2 s⁻¹ | 0; viskoziteye ek serbest hareket sönümü malzeme yasası olarak kullanılmıyor |
| Stone sıvı ν | Alan varsayılanı 1e-6 m²/s | Mevcut legacy Lava başlangıcı 0.5 m²/s; MSF'nin boyutsuz melt_viscosity değeri dönüştürülmedi |
| Stone sıvı yoğunluğu | Alan varsayılanı 1000 kg/m³ | Mevcut Stone yoğunluğu 2600 kg/m³ ile açık eşit-yoğunluk yaklaşımı |
| Stone viskoz çözüm ipuçları | Varsayılan 1 sweep / free slip | 24 sweep / no slip; mevcut Lava örneğiyle tutarlı |
| PE ve Stone soğuk ν | Sıcak ν'den düşük | max(eski soğuk ν, sıcak ν); soğurken incelme tersliği kaldırıldı, yeni ölçüm eğrisi iddiası yok |
| Gasoline flash Kelvin | 415.5 | 230.37; NIOSH -45°F dönüşümü |
| Alcohol flash / autoignition Kelvin | 391 / 545 | 285.93 / 638.15; mevcut bileşim verilerinin işaret ettiği etanol referansı |

SI fizik düzeltmeleri `BuiltinPresetPhysics.cpp` içinde, katalog yayınlanmadan
önce tek geçişte uygulanır. Eski büyük MaterialStateField.cpp yalnızca bu modülü
çağırır. Oradaki ilk kayıtlar ham girdidir; **etkin değerler finalizer sonrasıdır**.
Materyal kimlikleri, katı modeli ve DEM/MPM taşıyıcı tercihleri değiştirilmez.
Yüklenmiş proje override'ları ve domain'in kullanıcıya ait sayısal ayarları bu
fonksiyon tarafından değiştirilmez. Yeni preset solver ipuçlarını test etmek
için yeni domain veya malzemeyi yeniden seçme akışı kullanılmalı.

## Ortak doğrulama

Eriyebilir bir maddenin mevcut freeze-viscosity eğrisinde
`liquid_cold_viscosity >= liquid_kinematic_viscosity` olmalı. SubstanceLibrary
bu kuralı builtin ve türetilmiş maddelere uygular. UI, scripting ve IPC aynı
transactional patch işlemini kullandığından hatalı eğri tek anlamla reddedilir;
başka alanlar kısmen değişmez. Sıcak ν artırılırken gerekiyorsa soğuk ν aynı
patch içinde artırılmalı. Bu kural genel bir μ(T) yasası değil, bu freeze
yaklaşımının yön koşuludur.

## Katalog sadeleştirmesi

Altı birebir kopya kaldırıldı: Generic metal, Generic mineral, Generic polymer,
Generic fibrous solid, Generic liquid ve Generic mineral grains. Bu adlara
dayanan eski kayıtlar için alias/migrasyon eklenmedi; eski sahne uyumluluğu bu
çalışmanın hedefi değil.

`foundation` metadata tier'ı korunur, artık gerçek bir örneği başlangıç profili
olarak işaretler. 12 başlangıç örneği: Steel, Stone, Plastic (PE), Wood (Oak),
Water, Oil, Honey, Sand, Soil, Snow, Mud, Gelatin. Temsilci kopyası oluşturulmaz;
bu gerçek profillerden proje maddesi türetilir. Diğer 14 örnek arama/Details ile
erişilebilir; atanmış örnek her zaman seçicide görünür.

Mud/Slurry, Dough ve Chewing Gum aynı soft-material ailesinde gruplanır.
Iron, Copper, Paper, Cloth, Wax gibi ısıl/kimyasal farkları olan örnekler silinmez.
`substance.list`/`substance.get` metadata'sına `model_notes` eklendi; aynı notlar
UI tooltip'inde görünür. Generic adlı kullanıcı proje maddeleri yasaklanmadı.

## Açık eksikler

- Chocolate hâlâ molten constant-viscosity örneğidir. Akma eşiği ve shear thinning
  uygulanmadı; mevcut MPM EVP eşdeğer model gibi sunulmaz.
- Honey/Oil için sıcaklık ve bileşim bağımlılığı ölçülmüş bir yasayla kurulmadı.
- Oil kaba h değerlerinde hâlâ mevcut çözünürlük eşiğiyle atlanabilir; görünür
  rapor vardır, doğruluk/performans politikası bu değişiklikte değiştirilmedi.
- PE ve Stone, genel sınıflar için başlangıç örnekleri; belirli polimer veya
  magma numunesine ait fit değildir. Eşit Stone yoğunluğu da ölçüm değildir.
- Alcohol adı korunur; mevcut profil etanol olarak açıklanır. Yeni bir Ethanol
  kopyası eklenmez. Gasoline karışımdır; tek kaynama sıcaklığı mevcut yaklaşım
  olarak kalır. Parlama noktası ile kendiliğinden tutuşma sıcaklığı farklıdır;
  motorun mevcut combustion eşik/normalizasyon modeli yeniden yazılmadı.
- Eksik yasayı desteklemeden yeni “industrial chocolate”, “wet snow” veya
  “thick mud” kopyaları eklenmedi.

## Kaynaklar ve değerlerin kapsamı

- [NIOSH Ethyl alcohol](https://www.cdc.gov/niosh/npg/npgd0262.html): 55°F flash.
- [NOAA CAMEO Ethanol](https://cameochemicals.noaa.gov/chemical/667): 689°F autoignition;
  flash ve molekül kütlesi de mevcut etanol tanımıyla karşılaştırıldı.
- [NIOSH Gasoline](https://www.cdc.gov/niosh/npg/npgd0299.html): -45°F flash;
  yakıtın karmaşık karışım olduğu da belirtilir.

Yalnızca bu açık kaynaklı eşikler kaynak verisi düzeltmesidir. Viskozite, sweep,
yoğunluk yaklaşımı ve diğer katalog sayılarının tümü için ölçüm sertifikası
iddiası yapılmaz. °F → K: (F−32)×5/9+273.15.

## Derleme sonrası kontrol

1. Yeni domain'de Oil/Honey/Chocolate seç; internal friction0 ve mevcut sıvı
   modeli korunmuş olmalı. Yapışma ve duvar sürtünmesini ayrı tut.
2. Snow/Ice maddelerinin sıvı ν değerini Water ile karşılaştır; katı taşıyıcıları
   sırasıyla MPM/DEM olarak kalmalı. Erime sahnesinde kütle/enerji kabul
   ölçümlerini yeniden çalıştır; bu değişiklik sayısal yakınsama kanıtı değildir.
3. PE/Stone soğutma eğrisi sıcak ν'nin altına düşmemeli. Stone için ağır viskoz
   çözüm maliyetini gözle; preset artık su gibi düşük viskoziteli değil.
4. Pilotlu yakıt sahnelerinde Gasoline/Alcohol eşik düzeltmelerini kontrol et.
   Auto Ignite mevcut authored pilot seçeneğidir; fiziksel autoignition verisiyle
   aynı kontrol olarak yorumlanmamalı.
5. Harici testler (geçici proje maddeleri yaratıp kaldırır):
   `python scripts/test/rt_test_substance_profiles_ipc.py`
   ve `python scripts/test/rt_test_substance_catalogue_ipc.py`.
6. Kullanıcı C++ regresyonu: `builtin_preset_physics_test.cpp`,
   BuiltinPresetPhysics.cpp ve Math/Vec3.cpp ile bağlanır.

Codex derleme, shader compilation veya canlı uygulama doğrulaması yapmadı.
Statik yayın/entegrasyon kontrolleri fizik kabul testlerinin yerine geçmez.
