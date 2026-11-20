# Gizli duvar resmi — görsel deneme

2026-11-20: Açık Studio'daki workshop'tan ayrı Save As ile
`RayTrophiStudio/assets/scenes/hidden-mural-trial/hidden-mural-trial.rtp` üretildi.
Özgün workshop kaydı değiştirilmedi. Arka duvarda ağaç ve oturan tavşan var.

Bu bir motor özelliği değil, mevcut ortak script/IPC hizmetleriyle hazırlanan
sahne prototipidir. Yeni shader, C++ kodu veya build gerektirmez. Flat mesh
içe aktarma kullanılır; Triangle facade üzerinden geometri üretilmez.

## Mekanizma ve sınır

Duvarın 0,5 mm önünde, aynı taban renk ve roughness değerleri olan boyanmaz
bir ters şablon bulunur. Ağaç/tavşan açıklıkları alttaki gerçek wet-paint
yüzeyini gösterir. Boya değmemiş yerler gizli kalır. Boya rengi oyuncudan gelir.
Şablon 4.676 üçgen; collider veya ek paint atlası yoktur.

Bu yöntem şekil DIŞINDAKİ boyayı örter. Nihai tasarımdaki dışarıda normal boya,
içeride detaylı gizli resim davranışı henüz uygulanmadı. Göz/benek/yaprak detayı,
keşif yüzdesi ve ödül animasyonu yoktur. Şablon ray picking'i örtebilir; yalnız
görsel fizibilite içindir. Balon oyununun reset/progress kuralları değiştirilmedi;
ölçülen paint coverage alttaki örtülmüş boyayı da sayar.

## Kullanım

Ayrı terminalde, Studio açıkken:

```powershell
# Yalnız özgün workshop açık ve şablon henüz yokken:
python scripts/scenes/hidden_mural_trial.py install
# Kurulu denemede altı gerçek renkli boya darbesi:
python scripts/scenes/hidden_mural_trial.py preview
# Bütün oturum boyasını temizler; gizli şekiller tekrar kaybolur:
python scripts/scenes/hidden_mural_trial.py reset
# Şablonu sahneden kaldırır (kaydetmez):
python scripts/scenes/hidden_mural_trial.py remove
```

Studio'nun Paint wet reset işlemi de aynı çekirdeği kullanır. Islak boya oturum
verisidir; proje yeniden açılınca temiz başlar. Malzemeyi görmek için Material
görünümünü seçin. Canlı OBJ import sonrası bu denemede şablon görüntüye hemen
yansımadı; kurulum betiği kaydedilen denemeyi yeniden açıp önceki shading modunu
geri getirir.

Canlı kabul: temiz duvar, altı wet dab ile şekil sınırlarında kesilme ve
temizleme sonrası tekrar gizlenme görüntüyle kontrol edildi. C++ build ve
uygulama başlatma yapılmadı. Geçici doğrulama ekran görüntüleri silindi.
