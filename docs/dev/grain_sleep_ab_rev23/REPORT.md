# Revizyon 23 — uyku canlı A/B (2026-10-09)

Bu turda shader/C++ değişmedi, build veya uygulama açılışı yapılmadı. Açık uygulama
dış named-pipe IPC ile test edildi. Canlı revision 23 doğrulandı.

**Sonuç:** Dört repose kolunun açık/kapalı açı paritesi PASS. Uyku açık tam repose,
tane boyutu kapısında 4.04849° > 4° nedeniyle FAIL; kapalı tam repose PASS.
1.3M iki koşunun ölçek kapıları PASS. Tam uyku kabulü kapanmadı.

## Repose: dört kol, açık + kapalı

| Kol | Açık açı | Kapalı açı | Fark | ±2° paritesi |
|---|---:|---:|---:|---|
| mu_r_.05 | 17.93872° | 18.00762° | 0.06890° | PASS |
| mu_r_.3 | 34.45475° | 32.47735° | 1.97740° | PASS |
| mu_r_.1 | 20.00370° | 19.55552° | 0.44818° | PASS |
| mu_r_.1_r_.0175 | 24.05219° | 22.87855° | 1.17364° | PASS |

Her kolun kendi pile/floor/enerji/saçılma/kütle kapıları iki koşuda da PASS.
Düşük sürtünmeli açık saçılma 0.234667; önceki 0.317333 > 0.30 kırılması tekrarlanmadı.
Yüksek sürtünme açık/kapalı farkı 1.97740°; sınır içinde ama 2°'ye yakın.
Normal/küçük tane farkı açık **4.04848870° (FAIL)**,
kapalı **3.32303175° (PASS)**. Kapılar gevşetilmedi.
Açık test bu çapraz-kol assertion nedeniyle exit 1; kapalı exit 0.
Tam veriler `repose_comparison.json`, iki JSON ve log dosyasında.

## 1.3M saf DEM: 180 adım × 1/60 s, iki koşu

8 mm tane, 7434.8235 kg, 32258 yüzlü collider. Uyku modu her koşuda açıkça yazılıp
doğrulandı. Popülasyon >1M, held=false, enerji üst sınırı ve domain bounds PASS.
Bunlar bütün fizik/uyku kabulünün yerine geçmez. İki koşu aynı ayarlarla yeniden
doğdu; bit düzeyinde eşlenmiş state replay değildir. Başlangıç COM/enerji ve
kuyruk değerleri loglarda. GPU timestamp ölçümü ilk fizik adımından sonra etkin;
son pencereler iki koşuda da ölçümlü. `gpu_wait` host beklemesidir.

| Son 161–180 medyanı / 180. adım durum | Açık | Kapalı |
|---|---:|---:|
| Dış IPC duvar süresi | 5.750 s | 6.470 s |
| Host GPU wait | 4662.2 ms | 5388.5 ms |
| Substeps | 440 | 440 |
| Uyuyan tane (180) | 490007 (37.69%) | 0 |
| Kinetik enerji (180) | 0.06464599 J | 0.06668720 J |
| Öteleme RMS hız (180) | 4.170 mm/s | 4.235 mm/s |
| COM y (180) | 0.181048640 m | 0.181047566 m |

Son pencere duvar hızlanması **1.125x**.
Bu mevcut revision 23 ölçümüdür. İlk 90 adımda toplu erken uyku görülmedi;
90. adım açık sleeping=0, KE=155.77165 J; kapalı sleeping=0, KE=152.09686 J.
Kuyrukta görünen durgunluğa rağmen yalnız %37.7 uyuyor; kalan wake/eligibility
politikasının yeterli olduğu bu ölçümle gösterilmedi.

Native GPU snapshot farkları (son ~29 adım, gerçek kernel zamanı/çağrı):

| Kernel | Açık ms/çağrı | Kapalı ms/çağrı |
|---|---:|---:|
| sim_matter_grain_step | 10.3597 | 12.0847 |
| sim_matter_grain_list_build | 0.1930 | 0.1885 |
| sim_matter_grain_hash | 0.0188 | 0.0184 |
| sim_matter_grain_list_clear | 0.0476 | 0.0472 |

Tam çağrı sayıları ve snapshot'lar `scale_comparison.json` / `timing_*` içinde.
GPU profil ayarı önceki kapalı durumuna geri yüklendi. Repose'nin kendi H1 domain
ve source'ları silindi. Ölçek probu sözleşmesine uygun olarak Lim_Grain / Lim_Sand /
Lim_Floor sahnesi duraklatılmış ve uyku kapalı bırakıldı; kaydetme yapılmadı.

## Kullanıcı gözlemi: üst katman ve sıkışmış çekirdek

Kaynakta enerji/itki tabanlı wake yok. `g_fast && linked` hızdan wake bırakıyor;
`g_dynamic_contact` sönüm/sürtünme sonucundan önce göreli hareketten oluşuyor.
En somut karışıklık `balanced` içinde bu flag'in kuvvet/tork dengesiyle birleştirilmesi:
`!balanced && !g_fast` hareketli komşu veto'sunu da gerçek dengesiz destek gibi
komşulara yayıyor. Kuvveti dengeli bir tane bu yüzden ikinci halka wake tetikleyebilir.
Uyandırma tane düzeyindedir; tek hash hücresi flag'iyle tüm hücre uyandırılmaz.

[WAKE_AUDIT.md](WAKE_AUDIT.md) doğrulanan yolları ve fizik sözleşmesini içerir:
sönüm/sürtünme sonrası aktarılabilen net itki, hedefin kütle/ataleti ve destek
kapasitesi; biriken yavaş yük ve uyku denetimleri arasındaki yay geçmişi korunmalı.
Sadece büyük enerji eşiği koymak veya hızı artırmak uygulanmadı. Bu turda enerji
modeli kodlanmadı; ölçüm sırasında binary/source revision 23 kaldı.
