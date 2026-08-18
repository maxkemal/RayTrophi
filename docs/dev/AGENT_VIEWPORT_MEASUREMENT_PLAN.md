# Ajanın gördüğünü ölçebilmesi — viewport sürme ve render verisi

> **Durum:** AKTİF — Tek partilik ara adım. Node yönü ve ajan pipeline'ı bunun
> üstüne oturuyor; ikisi de görsel/performans doğrulamasını ajanın kendisinin
> yapabilmesine bağlı.

## Neden

CLAUDE.md'nin birinci kuralı: bir yetenek yalnızca panelden erişilebiliyorsa
**test edilemez** sayılır. Bugün (2026-08-16) siyah bant turunda zincirin
ortasında tam olarak böyle bir manuel adım vardı — ve bedeli ölçüldü:

- Viewport IPC'den sürülemediği için her kritik sayacı kullanıcı **panelden
  kopyalayarak** verdi. Üç tur böyle geçti.
- Bir tur tamamen boşa gitti: alınan dump **ateşsiz** bir kareden geliyordu
  (`density_samples = 4.470`) ve bunu ancak dump geldikten sonra fark ettik.
  Ajan kendi karesini sürebilseydi "alev oturana kadar bekle" koşulu
  ölçülebilirdi.
- Siyah bandın var/yok kararı **göz kararıyla** verildi. Oysa soru sayısaldı:
  *domain siluetinde parlaklığı 0,001'in altındaki piksel oranı nedir?*

★ Bugün ayrıca kanıtlandı ki bu boşluk yalnızca yavaşlatmıyor, **yanlış teşhis
ürettiriyor**: `fluid.get` bayat bir kayıt döndürdüğünde "termal zincir kapalı"
sonucuna varıldı. Ölçüm yolları güvenilir değilse ajan emin bir şekilde yanılır.

## Kural — çiğnenmeyecek

Bu partinin hiçbir maddesi IPC veri modeli sınırını zorlamaz. Bkz.
[IPC_SECURITY_PERFORMANCE.md](IPC_SECURITY_PERFORMANCE.md) "Data model boundary":
**yalnızca isim, id ve değer geçer.** Pointer/handle/çekirdek erişimi yok.

- Kare sürmek bir **komut**.
- Sayaç okumak bir **değer struct'ı**.
- Piksel okumak bir **tampon**.

★ `rt.ui` istisnası burada geçerli DEĞİL. O istisna **panel çizimini**
scriptlemeye dairdir; buradaki metotlar paneli değil **motoru** sürer. Bir çizim
çağrısı eklemiyoruz.

## Kapsam

Her metot dört dokunuşu tamamlar (çekirdek API + IPC dispatch + Python binding +
capability). Yeni namespace eklendiği için `scripts/audit_ipc_capabilities.py`
çalıştırılacak.

### 1. `viewport.render_frames { count }`

Viewport'u N birikim karesi sürer ve döner. Bugünkü tıkanma buydu: IPC'den sim
sürerken viewport boşa geçiyor ve `volume_rays = 0` okunuyor — bu bir ölçüm
değil, ölçüm **yokluğu**.

Dönüş: `samples_rendered`, `converged`, `ms_per_frame`.

### 2. `viewport.status`

`backend`, `samples`, `target_samples`, `ms_per_frame`, `width`, `height`.

Performans regresyonunu ajanın ölçebilmesi için gereken asgari yüzey. "Maliyet
patlaması" bugün ancak kullanıcının gözüyle rapor edilebiliyordu.

### 3. `render.probe { region?, threshold? }`

Son render'ın veya güncel viewport karesinin **sayısal** istatistiği:
`mean_luminance`, `min`, `max`, `black_fraction` (eşik altı piksel oranı),
`nan_fraction`, ve kaba bir histogram. İsteğe bağlı dikdörtgen bölge.

★ Bu, bu partinin asıl kazancı. Siyah bant soruşturmasının tamamı tek bir
sayıya indirgenebilirdi. `nan_fraction` ayrıca bugün hiç bakamadığımız bir
arıza sınıfını görünür kılar.

### 4. Offline render sayaçlarının yönlendirilmesi

`render.start` çalışıyor ama viewport backend'inin enstrüman tamponuna
yazmıyor; bugün 16 spp render alındı ve `volume_rays = 0` okundu. Render edilen
şey ölçülemiyorsa `render.start` bir doğrulama aracı değil, yalnızca bir çıktı
üreticisidir.

## Kabul ölçütü

Parti, ajan şu döngüyü **kullanıcıya hiç sormadan** kapatabildiğinde biter:

1. Sahneyi kur → `viewport.render_frames`
2. `render.probe` ile siyah piksel oranını ölç
3. `render.volume_stats` ile `density_samples / volume_rays` oranını ölç
4. İkisini de eşiğe göre geç/kal diye raporla

Bugünkü siyah bant turu bu döngüyle tek oturumda ve kullanıcı müdahalesi
olmadan kapanırdı.

## Sonrası

Bu bittiğinde sıradaki ana hat
[NODE_SIMULATION_ARCHITECTURE_PLAN.md](NODE_SIMULATION_ARCHITECTURE_PLAN.md);
[AGENT_PIPELINE_ARCHITECTURE.md](AGENT_PIPELINE_ARCHITECTURE.md) ise geri
alınamaz kimlik/veri düzeni kararlarını bu altyapı gerçek kullanımda
sınandıktan sonra verir.
