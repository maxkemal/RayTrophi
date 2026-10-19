# Fiziksel sentetik ses: tum olay aileleri icin kalite kapanisi

> **Durum:** AKTIF — 2026-10-19 kullanici onceligi. Kapsam tek bir dogal olayla
> sinirli degil; tum yapilar. Bu belge uygulama veya isitsel kabul tamamlandi demek degil.

Güncel belge sırası: önce [ana audio planı](AUDIO_MASTER_PLAN.md), ardından
[sentetik üretici teknik planı](PHYSICAL_AUDIO_SYNTHESIS_PLAN.md). Bu belge aileler
ve kalite hedefleri için kapsam referansıdır; ayrıntılı parti sırası bu iki plandadır.

## Hedef ve is sirasi

Kullanicinin asil hedefi Designer arayuzunden once, gercek doga olaylarindan kaliteli
sentetik ses uretilmesi. Kalan UI iyilestirmeleri bu hedefin onune gecmeyecek.
Tum audio islerini tek buyuk yamada kapatmak yerine tek teslim kapsaminda izlenen
ve kullanici build/dinleme kapilari bulunan asamalar uygulanacak. Tum ailelerde
ayni kalite omurgasi kullanilacak; ilk ailenin basarisi digerlerinin kabulunun yerine gecmez.

Mevcut PhysicalAudioDsp.cpp incelemesi: liquid_flow/gas_flow/friction/rolling,
filtrelenmis seeded gurultu ve yumusatilmis aktivite kullanir. Impact alti modlu
bir rezonator; contact_activity/deformation/fracture_activity ayni rezonatoru
surekli veya sentetik aralikli excitation ile uyarir. Bunlar kullanisli ilk
sonifikasyon modelleridir; tek basina dogal olaylarin ayrintili ses modeli degildir.

## Kapsam matrisi

| Aile | Mevcut temel | Kaliteyi kapatmak icin gereken ayrim |
|---|---|---|
| Sivi | liquid_flow aktivite dokusu | Akis/turbulans, damla-temas/sicrama, yuzey ve kap etkisi; kabarcik modeli ancak uygun veriyle |
| Gaz | gas_flow aktivite dokusu | Akis/turbulans, esinti ve jet/islik ayrimi; hiz ve geometri baglantisi |
| Kati temas | impact, contact_activity, friction, rolling | Tek darbe, surukleme ve yuvarlanmanin ayri zaman/enerji yapisi; iki malzeme, boyut ve rezonans |
| Deformasyon/kirilma | deformation, fracture_activity | Surekli deformasyon ile tekil kirilma olayinin ayrimi; olay zaman/enerji/provenansi |
| Atmosfer | Mevcut ruzgar/yagmur/thunder asset ve tepki yollari | Sentetik olay modeli ile kayit tabanli kaynak ayrimi; yagis siddeti/yuzey, ruzgar hamlesi, yildirim-gecikme baglantisi |
| Termal olaylar | Bu fiziksel generator listesindeki ayri bir model degil | Kaynama/buhar/yanma gibi olaylar icin model ve olculen kontrol sozlesmesi; sadece sicakliktan uydurma ses olayi cikarmama |

Malzeme ailesi uygun bir baslangic preset'i saglayabilir. Metal/organik gibi tek
etiket; temas cifti, boyut, rezonans ve sonumun yerine gecmez. UI sade aileler sunar;
modeller gerekli sayisal fiziksel kontrolleri ortak Substance/override baglamindan
acikca alir. PBR gorunumunden akustik parametre tahmin edilmez.

## Tek ortak zincir

1. Solver-owned veri: hangi domain/nesne, hangi olcu ve birim, kaynak/provenans,
   zaman, konum, olay kimligi; olculmeyen veri olculdu diye sunulmaz.
2. Surekli kontrol ve tekil olay ayri sozlesmeler: aktivite degisiminden otomatik
   collision/splash/bubble olayi uydurulmaz. Olaylarin kapasite/overflow/replay
   davranisi tanimlanir. Diger ajanin birlesik domain kodu ile entegrasyon arayuzu
   bu veri hazirligina gore yapilir; bu calisma paralel solver degisikliklerini ezmez.
3. Olay ailesine uygun, deterministik ses cekirdegi: doku, transient ve rezonans
   gerektigi yerde ayri uretilir. Mevcut tarifler sessizce degistirilmez; DSP sesini
   degistiren revizyon tarif surumuyle izlenir. Parametreler callback disinda hazirlanir.
4. Mevcut ortak uzamsal/akustik zincir: mesafe, dogrudan yol, malzeme gecisi ve
   yansimalar. Generator icine ikinci bir sahne akustigi eklenmez.
5. Ayni tarif/seed/kontrol-olay gecmisiyle canli calisma, replay ve offline uretim.
   Mevcut PhysicalAudioHistory/Schedule/Timeline/Bake bu hedefin temelidir;
   dry bake bulunmasi tam akustik export esligi veya exact solver kaydi demek degil.
6. Mevcut UI/Python/IPC ayni servisi kullanir. Yeni kontrol ve durumlar tum yuzeylerde
   gorunur; ayri UI-only DSP, olay listesi veya gecmis otoritesi olusturulmaz.

## Her aile icin tamamlanma kosulu

- Dusuk/orta/yuksek aktivite; ani baslangic/durus; tekrarlanan ve ust uste olaylar.
- Ses timbresi ve zamansal davranis, yalniz ses yuksekligi degil, olayla uyumlu.
- Sifir aktivitede sessizlik ve dogru kuyruk; klik/NaN/clip/tekrarlanan olay yok.
- Farkli malzeme/yuzey/boyutlarda anlamli, acikca modellenmis farklar.
- Ayni tarif/seed/gecmis ayni sonucu verir; callback blok boyutu davranisi bozmaz.
- Pause/seek/replay, kaynak degisimi ve scene reset eski olaylari yeniden calmamali.
- Proje kaydet/ac, dosya/asset/klip baglantisi; dry ve akustik export kapsami acik.
- Eszamanli 1/8/16/32 kaynakta callback butcesi, underrun ve bellek olculur.
- Referans sahnelerde kullanici dinleme kabulu: akis/sicrama, ruzgar/gaz,
  darbe/surtunme/yuvarlanma, deformasyon/kirilma ve atmosfer/termal olaylar ayri.
  PCM enerjisi veya guzel FFT goruntusu tek basina kalite kaniti degildir.

Tam fiziksel akustik radyasyon/SPL veya tum doga olaylarinda gercege esdegerlik
vaat edilmez. Hangi aile artistik procedural yaklasim, hangisi olculmus olayla
suruluyor, hangisinde veri yok, acik gorunur. Kalite; model, fizik baglantisi,
zamansal tutarlilik ve dinleme kabulunun birlikte sonucudur.

## Ilk uygulama kapisi ve kalan audio isleri

Ilk ortak is: mevcut sekiz generatorun tarif/DSP/olay-gecmisi/akustik-bake
sinirlarini bu matrise gore envanterlemek, her ailenin gerekli kontrol veri
sozlesmesini belirlemek ve ortak surumlu kalite mimarisini uygulamak. Eksik
solver olaylari hazir degilse acik artistik model olarak kalir; sahte fiziksel
olcumle tamamlandi sayilmaz. Kullanici dogrulamasindan once tum aileleri etkileyen
varsayilan ses degisiklikleri veya eski projelerin sessiz yeniden yorumlanmasi yok.

Onceki Designer/EQ/Analyze teslimleri korunur. Odak temelli Undo/Redo, islenmis
asseti mevcut klibe remap, streaming, ayri direct/reflected/transmitted enerji
analizi ve genel akustik kapanisi ayri kalan maddelerdir. Bunlar dogal ses kalitesini
tek basina cozen isler degildir ve bu belgeyle tamamlandi sayilmaz.

Codex derleme, shader compilation veya uygulama baslatma yapmaz. Canli kabul
kullanici build'inden sonra dis surecten scripts/test/rt_ipc.py ile ve dinlemeyle.

Ilgili kaynaklar: PHYSICAL_AUDIO_GENERATION.md, PHYSICAL_AUDIO_BAKE.md,
PHYSICAL_AUDIO_STORED_HISTORY.md, AUDIO_FLUID_SYNTHESIS_PLAN.md,
SCENE_ACOUSTICS_ROADMAP.md, AUDIO_DESIGN_CONTEXT_WORKFLOW_PLAN.md.
