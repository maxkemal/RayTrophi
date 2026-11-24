# Lenovo TB330FU Android Player 2257 ölçümü

2026-11-23. Android 15, Mali-G52 MC2, arm64 Release, Gizli Oyuncaklar (workshop), Jolt açık. Fiziksel ekran 1920×1200 yatay. İç çözünürlük 960×600, render scale 0,5. Swapchain doğrudan sunum, yakalama kapalı. Sahne 27 draw, 39.934 görünür üçgen, 8 boya yüzeyi ve 8.192 boya tile’ı.

## İlk oyun karesine kadar yükleme

Tek soğuk uygulama açılışında sahne yükleme ölçerinin toplamı **46,88 s**. Listelenen adımların toplamı **46,837 s**. Pencere açılışı ve ilk Vulkan aygıt kurulumu sahne yükleme ölçerinden önce gerçekleşti. Bu yüzden tabloda prepare_device yalnız mevcut backend’i alma süresidir.

| Sıra | Aşama | Ayrıntı | ms |
|---:|---|---|---:|
| 1 | `check_project` | — | 72,561 |
| 2 | `prepare_device` | — | 0,272 |
| 3 | `project_data` | — | 0,194 |
| 4 | `project_data` | Opening project file with turbo parser... | 87,199 |
| 5 | `project_data` | Clearing scene... | 18,699 |
| 6 | `read_geometry` | Loading geometry... | 14,118 |
| 7 | `read_textures` | Restoring textures... | 1,288 |
| 8 | `read_materials` | Loading materials... | 191,824 |
| 9 | `project_data` | Loading lights... | 0,403 |
| 10 | `project_data` | Loading cameras... | 0,329 |
| 11 | `project_data` | Loading render settings... | 205,909 |
| 12 | `project_data` | Loading terrain system... | 0,294 |
| 13 | `project_data` | Loading river system... | 0,170 |
| 14 | `project_data` | Loading foliage system... | 0,337 |
| 15 | `project_data` | Loading mesh modifiers... | 0,010 |
| 16 | `project_data` | Loading geometry node graphs... | 0,056 |
| 17 | `project_data` | Loading paint layers... | 367,297 |
| 18 | `restore_audio` | Restoring audio... | 2,797 |
| 19 | `prepare_particles` | Preparing particle systems... | 22575,964 |
| 20 | `restore_physics` | Restoring physics... | 0,307 |
| 21 | `project_data` | Loading hair system... | 21,072 |
| 22 | `project_data` | Rebuilding model contexts... | 0,056 |
| 23 | `project_data` | Project data loaded, finalizing... | 0,051 |
| 24 | `game_setup` | — | 0,002 |
| 25 | `scene_setup` | — | 0,086 |
| 26 | `upload_materials` | — | 126,371 |
| 27 | `prepare_viewport` | — | 1500,947 |
| 28 | `game_design` | — | 21596,664 |
| 29 | `first_frame` | applied: 8 paint surface(s), balloon demo prepared | 51,419 |

`prepare_particles` **22,58 s** ve `game_design` **21,60 s**, birlikte **44,17 s (%94,2)**. `prepare_viewport` **1,50 s**. Demo ayrıca **1,586 s prepare_ms** ve **0,505 s atlas_ms** raporladı. Bu iç sayaçlar yükleme toplamına eklenmez. Havuz hazırlığı, SDF beklemesi ve başlatıcı arkasındaki karelerin bu iki üst aşamadaki payı ayrı ölçülmedi.

## Oyun renderi ve sunum

Boş ve tek patlama sonrası ilk pencerelerde compute zamanlaması kapalıydı. Üçüncü pencere aynı patlamadan sonra boya devam ederken compute timestamp’leri açık alındı. Patlama 5.110 damla üretti. Pencereler yaklaşık 11 saniyelik kısa örneklerdir. `player.display.frame` değerleri her pencerenin sonunda okunan son kayan pencereye aittir.

| Metrik | Boşta | Patlama sonrası | Boya, compute timestamp açık |
|---|---:|---:|---:|
| Duvar saati (s) | 11,22 | 11,83 | 11,43 |
| GPU örneği olan raster kare | 17 | 13 | 11 |
| Raster GPU ort. (ms/kare) | 696,68 | 722,86 | 734,88 |
| Raster GPU p95 (ms/kare) | 709,43 | 727,73 | 740,68 |
| Raster CPU kayıt ort. (ms/kare) | 2,45 | 2,95 | 2,86 |
| Raster CPU kayıt p95 (ms/kare) | 3,91 | 3,23 | 3,33 |
| Player sunulan FPS | 1,40 | 0,83 | 0,88 |
| Player advance (ms/iterasyon) | 4,75 | 442,09 | 1126,98 |
| Player present çağrısı (ms) | 702,44 | 756,85 | 2,03 |
| Player sunum aralığı ort. (ms) | 713,20 | 1204,81 | 1134,41 |
| Raster slot bekleme, son kare (ms) | 0,006 | 0,008 | 0,006 |
| Raster submit, son kare (ms) | 0,314 | 0,408 | 0,563 |
| Görüntü GPU→CPU readback, son kare (ms) | 0,000 | 0,000 | 0,000 |
| Raster host read, son kare (ms) | 0,000307 | 0,000307 | 0,000308 |
| Aktif boya tile’ı, son tick | 0 | 980 | 932 |
| Damgalanan texel, son tick | 0 | 54753 | 17409 |

Player `present` çağrısı kuyruk beklemesini de kapsar. GPU raster süresiyle toplanmaz. Compute timestamp açılınca beklemenin büyük kısmı `present` yerine `advance` içinde görünüyor.

### Raster geçişlerinin tamamı

Her hücre GPU ort. / GPU p95 / CPU kayıt ort. (ms), sonra geçişin çalıştığı kare sayısı. Çalışmayan geçiş — ile gösterildi. Mali TBDR, HDR render pass’in GPU süresini ilk `sky` damgasına yazıyor. `sky` ve `main_pass` GPU sütunları birlikte okunmalı. Bu ölçümden HDR pass içindeki sky, depth ve opaque GPU maliyetleri tek tek çıkarılamaz.

| Geçiş | Boşta | Patlama sonrası | Boya, compute timestamp açık |
|---|---:|---:|---:|
| `gpu_cull` | 0,222 / 0,609 / 0,091 (17) | 0,721 / 1,106 / 0,092 (13) | 0,906 / 0,992 / 0,096 (11) |
| `shadow_atlas` | 37,792 / 40,261 / 1,532 (17) | 35,911 / 37,803 / 1,567 (13) | 35,819 / 35,954 / 1,542 (11) |
| `transmission_prep` | 0,018 / 0,025 / 0,033 (17) | 0,019 / 0,023 / 0,043 (13) | 0,020 / 0,026 / 0,051 (11) |
| `sky` | 423,711 / 428,116 / 0,069 (17) | 434,690 / 436,040 / 0,086 (13) | 438,263 / 440,660 / 0,094 (11) |
| `depth_prepass` | 0,002 / 0,008 / 0,112 (17) | 0,002 / 0,003 / 0,124 (13) | 0,001 / 0,004 / 0,115 (11) |
| `rt_shadow` | — | — | — |
| `screen_gi_trace` | — | — | — |
| `screen_gi_filter` | — | — | — |
| `main_pass` | 0,001 / 0,003 / 0,161 (17) | 0,001 / 0,004 / 0,165 (13) | 0,002 / 0,004 / 0,169 (11) |
| `volume_sdf` | — | — | — |
| `reflection` | — | — | — |
| `reflection_filter` | — | — | — |
| `transmission` | 222,284 / 232,008 / 0,253 (17) | 239,148 / 246,894 / 0,317 (13) | 247,007 / 252,324 / 0,317 (11) |
| `taa` | 5,288 / 5,451 / 0,060 (17) | 4,674 / 5,572 / 0,067 (13) | 5,313 / 5,461 / 0,080 (11) |
| `post` | 6,941 / 7,016 / 0,035 (17) | 6,930 / 6,961 / 0,040 (13) | 6,937 / 6,961 / 0,047 (11) |
| `overlay` | 0,407 / 0,441 / 0,045 (17) | 0,752 / 0,830 / 0,395 (13) | 0,600 / 0,711 / 0,287 (11) |

### Simülasyon ve billboard CPU ölçümleri

Hücreler örnek sayısı / toplam pencere ms / çağrı ortalaması ms. Simülasyon alt sayaçları üst tick ile örtüşür. GPU saatleri ve CPU bekleme süreleri toplanmaz.

| Sayaç | Boşta | Patlama sonrası | Boya, compute timestamp açık |
|---|---:|---:|---:|
| `player.advance.simulation_tick` | 22 / 72,806 / 3,309 | 18 / 4771,189 / 265,066 | 16 / 18059,396 / 1128,712 |
| `simulation.balloon_demo` | 22 / 59,607 / 2,709 | 18 / 73,794 / 4,100 | 17 / 42,113 / 2,477 |
| `simulation.timeline` | 22 / 6,248 / 0,284 | 18 / 283,263 / 15,737 | 16 / 116,367 / 7,273 |
| `simulation.wet_paint` | 22 / 1,744 / 0,079 | 18 / 4409,088 / 244,949 | 16 / 17899,337 / 1118,709 |
| `simulation.compute_submit` | 22 / 0,058 / 0,003 | 18 / 0,332 / 0,018 | 16 / 0,227 / 0,014 |
| `paint.wet.stats_collect` | 22 / 0,040 / 0,002 | 18 / 0,395 / 0,022 | 16 / 0,522 / 0,033 |
| `viewport.particle_billboards.build` | 22 / 0,930 / 0,042 | 18 / 15,974 / 0,887 | 16 / 10,072 / 0,630 |
| `viewport.particle_billboards.drain` | — | 2 / 1388,299 / 694,149 | — |
| `viewport.particle_billboards.upload` | — | 3 / 1,022 / 0,341 | — |
| `viewport.particle_billboards.vertex_write` | — | 18 / 4,688 / 0,260 | 16 / 2,186 / 0,137 |

### Islak boya ve parçacık GPU zaman çizelgesi

Aşağıdaki örnekler üçüncü pencereden. `paint_wet` alt modların toplamıdır. Ayrı compute bağlamı bu sahnede etkin değildi. Tabloyu üretmek için compute timestamp kullanıldı.

| Sayaç | Dispatch | Toplam GPU ms | Ort. ms/dispatch | P95 ms/dispatch | Tepe ms |
|---|---:|---:|---:|---:|---:|
| `paint_wet` | 83 | 4166,806 | 50,202 | 178,556 | 181,951 |
| `paint_wet.tick` | 11 | 1919,353 | 174,487 | 181,951 | 181,951 |
| `paint_wet.channels` | 11 | 1058,267 | 96,206 | 101,797 | 101,797 |
| `paint_wet.stamp` | 11 | 493,320 | 44,847 | 62,725 | 62,725 |
| `paint_wet.coverage` | 6 | 427,717 | 71,286 | 72,588 | 72,588 |
| `paint_wet.resolve` | 11 | 265,378 | 24,125 | 24,930 | 24,930 |
| `paint_wet.channels_compact` | 11 | 2,254 | 0,205 | 0,208 | 0,208 |
| `sim_particle_collide` | 11 | 1,363 | 0,124 | 0,155 | 0,155 |
| `paint_wet.rotate` | 11 | 0,383 | 0,035 | 0,036 | 0,036 |
| `sim_particle_ballistic` | 11 | 0,382 | 0,035 | 0,039 | 0,039 |
| `paint_wet.consume` | 11 | 0,135 | 0,012 | 0,013 | 0,013 |

## Darboğaz sırası

- Açılışta ilk ayrım, `prepare_particles` havuz kurulumunun ve `game_design` içindeki SDF/atlas hazırlığı ile beklenen karelerin ayrı süreleri.
- Oyun boşta başlıca GPU yükü birleşik HDR geçişi, ardından transmission ve gölge atlası. Doğrudan swapchain ile görüntü readback’i 0 ms.
- Islak boyada ilk hedef `tick`, ardından `channels` ve `stamp`. `channels_compact` ayrı ölçüldü. Kare hızı testi compute timestamp kapalı pencereden okunmalı.

Ham dosyalar yerel `tools/perception_bridge/out/` altında. Sahne şekli ve arka plandan dönüş `tablet_2257_swapchain_acceptance_20261123.json`, IPC sahne zaman çizelgesi, raster örnekleri, sunum ve compute verisi `tablet_2257_stage_profile_20261123.json`.
