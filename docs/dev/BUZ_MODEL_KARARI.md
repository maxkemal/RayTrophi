# Katı buz modeli kararı: granül mü, rijit gövde mi?

> **Durum:** AKTİF — 2026-10-08. Karar önerisi; kullanıcı onayı bekleniyor. Kod yazılmadı.

## Soru

Donmuş (263 K) su ya da buz, sıvı domain içinde nasıl davranmalı? Döküldüğünde katı parçalar halinde
düşmeli, yığılmalı ve çarpınca kırılmalı (sıvı gibi akmamalı). Bu parçalar üç yoldan birinde çözülebilir:

| Seçenek | Yol | Mevcut altyapı | Güçlü yan | Zayıf yan |
|---|---|---|---|---|
| **A. Granül taşıyıcı (DEM küre)** | H1 grain yolu (`MatterGrain*`) | Var, GPU/CPU referansı kanıtlı | Ayrı parçaların döküm, yığılma, çarpma davranışı doğal; temas/sürtünme/yuvarlanma var | Kürelerle yaklaşıklık; buz kırılması yok |
| **B. Rijit gövde** | Rigid-body çözücü (ayrı) | Var, ama fluid domain ile çift yönlü bağ YOK | Parça şekli ve dönme doğru | Fluid–rijit bağlama yeni iş; maliyet ve sahiplik sorunu |
| **C. Elastik MPM sürekli ortam** | `Elastic` model yolu | Var | Tek çözücü; sürekli deformasyon | Kırılma/hasar yok; yığılma için sürtünme/yoğunluk modeli yok; statik katı için uygun, dökülme için değil |
| (Reddedilen) Drucker-Prager MPM ("kum gibi") | `Granular` MPM | Var | — | Buz sürtünme açısı 35° (kum varsayılanı) — fiziksel değil |

## Bulgu: Ice preset bugün kum gibi çözülüyor

`MaterialStateField.cpp:617` içindeki "Ice" maddesi granül alanlarını ayarlamıyor. Varsayılanlar devreye giriyor:

- `granular_friction_degrees = 35` (kum)
- `granular_young_modulus = 2.0e5 Pa` (buzun ~9.3 GPa değerinin yaklaşık 45 000 kat altı)
- `default_constitutive_model` = `Fluid` (ayarlı değil)

Yani bugün "Ice" maddesi granül olarak çözülseydi, yumuşak bir kum yığını gibi davranırdı. Bu,
yüksek olasılıkla sessiz bir hata: sonuç makul görünür, kimse fark etmez. Karar ne olursa olsun bu
alanlar açıkça ayarlanmalı.

## Kaynaklı parametreler (ve boşluklar)

Kaynaklar bu notta yalnızca ikincil veya sınırlı bağlamda kullanılmıştır; her değer tek bir testten
değil, farklı sıcaklık ve numune koşullarından gelir.

| Parametre | Değer | Kaynak | Not |
|---|---|---|---|
| Young modülü (polikristal, 263 K) | 9.3 GPa | [arXiv 2205.05219](https://arxiv.org/pdf/2205.05219) (Schulson & Duval 2009 atfı) | Sıcaklıktan belirgin etkilenmiyor (Mellor 1975 atfı) |
| Poisson oranı | 0.32 | [arXiv 2205.05219](https://arxiv.org/pdf/2205.05219) | Aynı atıf |
| Yoğunluk | 900 kg/m³ tablo değeri; 917 kg/m³ standart | [Engineering ToolBox](https://engineeringtoolbox.com/ice-properties-mechanical-d_2179.html) (kısmi) | 917 için doğrudan kaynak bulunamadı |
| Basınç dayanımı | 6 MPa | [Engineering ToolBox](https://engineeringtoolbox.com/ice-properties-mechanical-d_2179.html) | Test koşulu belirtilmemiş |
| Çekme dayanımı | 1 MPa | [Engineering ToolBox](https://engineeringtoolbox.com/ice-properties-mechanical-d_2179.html) | Aynı |
| Kırılma tokluğu K_IC | 0.12 MPa·m^½ | [Engineering ToolBox](https://engineeringtoolbox.com/ice-properties-mechanical-d_2179.html) | İkincil kaynak |
| Buz–buz kinetik sürtünme | ~0.017–0.06 (dolaylı) | [Hypertextbook](https://hypertextbook.com/facts/2004/GennaAbleman.shtml) (curling 0.0168, patinaj 0.0046–0.0059); [Journal of Glaciology](https://resolve-he.cambridge.org/core/journals/journal-of-glaciology/article/on-friction-and-surface-cracking-during-sliding-of-ice-on-ice/9DC7B4860888F042A475D2BB5055271B) (sayılar metinde yok) | Ayrıntılı sayılar tam makalede aranmalı |
| Buz–buz **statik** sürtünme | **bulunamadı** | — | Bowden atfı yalnızca kayak vaksı için 0.24; buz–buz değil |
| Granül buz AOR (repose) | **bulunamadı** | — | Kar DEM kalibrasyonu 34° (sürtünme 0.3, yuvarlanma 0.2) verir ama kar, buz değil; [arXiv 2007.01694](https://arxiv.org/pdf/2007.01694) |
| Restitüsyon (çarpışma) | **bulunamadı** | — | Mevcut grain yolu restitüsyonu parametre olarak alıyor |

**Kısa sonuç:** Buz–buz sürtünmesinin büyüklük sırası yaklaşık 0.02–0.06 (kum: ~0.6). Kum varsayılanı (35°)
bu yüzden buz için yanlış. Statik sürtünme ve AOR için ölçüm gerekiyor; uydurmayacağız.

## Öneri: A (granül taşıyıcı) — ilk aşama; B için kapı açık

1. **Gerekçe.** Döküm, yığılma ve çarpma davranışı, ayrı parçacıkların birbirinden kopuk bir topluluk olmasını
   gerektirir. Bu, H1 grain yolunun zaten yaptığı şeydir. Dökülen buz parçaları birbirine yapışmadığı sürece
   (kohezyon sıfır) sürekli ortam (MPM) yanlıştır; rijit gövde ise ancak parça sayısı az olduğunda uygundur.
2. **Parametreler (A için):** yoğunluk 917 kg/m³; sürtünme ve restitüsyon için **ölçülmüş değer gelene kadar**
   buz–buz kinetik μ aralığının ortası (~0.04) ile başlanır ve bu açıkça "geçici" işaretlenir; E/ν yalnızca
   sürekli ortam yolu için kullanılır, DEM temas sertliği numerik olarak ayrı ayarlanır.
3. **Bağlanacak kapılar:** (a) ice preset'inde granül alanları açıkça ayarlansın (sürtünme, E, ν), (b)
   statik buz–buz sürtünmesi ve AOR bir ölçümle doğrulansın, (c) kırılma/hasar bu aşamada yok — "katı
   parça kırılır" iddiası yapılmaz.
4. **B (rijit gövde) ne zaman?** Yalnızca büyük, tek parça buz blokları (ör. buz kütlesi çarpması)
   gerekirse. Fluid–rijit çift yönlü bağ yeni iş olduğundan şimdi yapılmaz.
5. **C (elastik MPM) ne zaman?** Statik, deforme olmayan katı buz (sabit blok) için; dökülme için değil.
   Statik katı bloklar zaten `solid_substance_tags` yolunda, bu yol değişmez.

## Kullanıcıdan gereken onay

- [ ] Seçenek A (granül taşıyıcı) ilk aşama olarak onaylansın mı?
- [ ] Ice preset'inde granül alanlarının açıkça ayarlanması (ilk iş) onaylansın mı?
- [ ] Statik sürtünme ve AOR için ölçüm: hangi sahne ile yapılsın? (Önerim: tek buz parçası ve yığın dökümü, H1 grain sahnesi.)

## Açık

- Kaynakların tam makaleleri (Journal of Glaciology sayıları, Schulson & Duval) okunmadı.
- Buz parçası boyutu/parça sayısı için grain yarıçapı ve domain çözünürlüğü seçimi (T2b-3 tasarımında).
