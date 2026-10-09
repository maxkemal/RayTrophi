# DEM uyku A/B — 2026-10-09

Canlı uygulama **revision 22**. Kaynak **revision 23**, derlenmedi. Testler açık
uygulamaya dış named-pipe IPC ile koşuldu; ajan build/uygulama açılışı yapmadı.

## 1.3M saf DEM, 180 adet 1/60 s adım

8 mm tane, 7434.8235 kg, 32258 yüzlü collider. İki koşuda population/held/enerji
üst sınırı/domain bounds kapıları PASS; bunlar repose/fizik paritesi kabulü değil.
Başlangıç doğum örnekleri çok yakın ama bit düzeyinde aynı değil: COM farkı
~5.1e-6 m. Aynı ayarlarla yeniden oluşturulan sahne A/B'sidir. Son pencerelerde
iki koşunun da GPU timestamp ölçümü açık. `gpu_wait` ayrı host sayacıdır.

| Son 161–180 medyanı / 180. adım durum | Uyku açık | Uyku kapalı |
|---|---:|---:|
| Dış IPC duvar süresi | 5.225 s | 6.530 s |
| Host GPU wait | 4660.9 ms | 5451.4 ms |
| Substeps | 440 | 440 |
| Uyuyan tane (180) | 420072 (32.31%) | 0 |
| Kinetik enerji (180) | 0.06272990 J | 0.05736850 J |
| Öteleme RMS hız (180) | 4.108 mm/s | 3.928 mm/s |
| COM y (180) | 0.181130106 m | 0.181034101 m |

Son pencere duvar hızlanması **1.250x**. Bu revision 22
ölçümü; revision 23 kazancı henüz ölçülmedi. İlk 81–90 medyanları 7.12 s açık /
6.82 s kapalı; 90'da yalnız 30/1.3M tane uyuyordu. Enerjiler 155.0077 / 153.7088 J.
Görsel durgunluk tek başına uyku kanıtı değil.

Native GPU snapshot farkları (son ~29 adım, kernel çağrısına normalize):

| Kernel | Açık ms/çağrı | Kapalı ms/çağrı |
|---|---:|---:|
| sim_matter_grain_step | 10.2735 | 11.8587 |
| sim_matter_grain_list_clear | 0.2807 | 0.2818 |
| sim_matter_grain_list_build | 0.1417 | 0.1406 |
| sim_matter_grain_hash | 0.0530 | 0.0532 |

Snapshot/call sayıları `scale_comparison.json` ve `timing_*` kayıtlarında.
Profil ayarı önceki kapalı durumuna geri yüklendi. Ölçek probunun sözleşmesine
uygun Lim_Grain / Lim_Sand / Lim_Floor sahnesi duraklatılmış, uyku kapalı bırakıldı;
kaydetme yapılmadı.

## Repose: kabul açık

1500 taneli iki kol, revision 22:
- mu_r_.05: açı 22.41924° açık / 22.13641° kapalı; açı paritesi PASS ama açık
  saçılma 0.317333 > 0.30: fizik FAIL (kapalı 0.255333).
- mu_r_.3: açı 30.03675° açık / 32.83456° kapalı; fark 2.79781° > 2°: FAIL.
- Açık GPU wait tail medyanları 8.390 / 5.368 ms; kapalı 63.942 / 58.045 ms.
  Fizik FAIL olduğundan hızlanma tam kabul sayılmaz. Kalan iki kol ve
  motion/coexist/revision 23 regresyonları yeniden koşulmalı.

## Kaynak revision 23

Hızlı/dengesiz tane zaten uyanık komşuya sürekli WAKE bırakıp rest sayacını
sıfırlıyordu. Artık coherent okuma ile yalnız uyuyan komşuya atomik wake yazılır.
Uyanık aday kendi temas hızı/kuvvet/tork kapısını tam çözer. Pozitif rest CAS
concurrent WAKE'i korur; geçici rest=0 yarışı geri gelmez. Sıfır rest ve wake
okumasında gereksiz atomikler, uyku kapalıda rest atomikleri/denge hesabı atlanır.
2 mm/s ve 0.2 s varsayılan eşikler değiştirilmedi. 17 descriptor / 128 B push /
4 dispatch-substep korunur. 9 derlemesiz kaynak kontrolü PASS; altı düzenlenen
Python dosyasının AST ve iki script kopyası eşitliği PASS. GPU kabulü henüz yok.
Sonraki toplu shader+C++ build'de revision 23 birlikte gelmeli; canlı sleep probu
eski revision 22 binary'sini artık kabul etmez. Sıra NEXT_BUILD_CHECKS sonunda.

## DEM + granular MPM

Sand DEM + Soil MPM aynı Matter domain'de `--owners-only` PASS: owner 1+1,
17 temas, impulse 0.13090269 Ns, residual 1.16415e-10 Ns; ortak saat 130 transport /
7 continuum grid adımı. Karışık saatte uyku koruma amacıyla kapalı tutuluyor.
Elastik MPM veya üç-sahipli sıvı kabulü değildir: varsayılan tam probda 12 örnekte
liquid support aktif olmadı, FAIL. Teşhis açık; tam probun FAIL nedeni bu paragrafta kayıtlı.
`dem_mpm_owners.log` ve `repose_comparison.json` ayrıntıları içerir.
