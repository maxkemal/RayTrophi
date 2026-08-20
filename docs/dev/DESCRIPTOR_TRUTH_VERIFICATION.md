# Descriptor'lar doğru mu söylüyor — iddia denetimi

> **Durum:** REFERANS — 2026-08-19'da kuruldu ve koşuldu. `agent.discover`'ın
> raporladığı kapsam metriğinin ne ölçtüğünü, ve ne ölçmediğini tanımlar.

## Sorun

`documented_coverage: 1.0` **"özet yazılmış"** demek. **"Özet doğru"** demek
değil. 311 metodun açıklaması var ve hiçbiri kontrol edilmemişti.

Bu, [AGENT_DISCOVERY_LAYER_PLAN.md](AGENT_DISCOVERY_LAYER_PLAN.md)'daki 299 boş
kaydın bir kat yukarısı. Orada alet **varlık** ölçüyordu ve **doluluk** diye
okunuyordu; burada alet **doluluk** ölçüyor ve **doğruluk** diye okunuyor. Aynı
sınıf, daha sinsi hâli — çünkü bu sefer kayıtlar gerçekten dolu.

## Ne denetlenebilir

Nesir genel olarak doğrulanamaz. Ama bu notların **iddia ettiği** şeyin çoğu
mekanik:

| İddia | Kontrol |
|---|---|
| Not `voxel_size`'dan bahsediyor | O metodun (veya işaret ettiği metodun) parametresi olmalı |
| Not `nan_fraction`'dan bahsediyor | Motor gerçekten o alanı döndürmeli (`--live`) |
| `verify_with: X` | X **yazan** bir metot olamaz — mutasyon doğrulama yapmaz |
| `invalidates: X` | X kapalı bir durum sözlüğünde olmalı (`sim_cache` ≠ `simulation_cache`) |
| Not `gas.set_shader`'dan bahsediyor | O metot dispatch edilmiş olmalı |
| Enum değeri | Şemada ilan edilmişse topraklanmış sayılır |

Geriye kalan **topraklanmamış iddia**dır: nesir, bu build'de olmayan bir şeyi
adlandırıyor. Bunlar `scripts/descriptor_claim_baseline.json`'da tutulur, yani
sayı **yalnızca düşebilir**. Bir yeniden adlandırma bir notu sessizce
geçersizleştirdiğinde yakalanan şey budur.

```powershell
python scripts/verify_descriptor_claims.py          # statik
python scripts/verify_descriptor_claims.py --live   # motora da sorar
python scripts/verify_descriptor_claims.py --accept # taban çizgisini bilerek tazele
```

## İlk koşunun bulduğu (2026-08-19)

Başlangıç: **%56 topraklanmış**. Bitiş: **%97.7**. Aradaki fark üç gerçek arıza
ve iki tokenizer yanlış pozitifiydi.

1. ★★ **`agent.discover`'ın kendi notu yalan söylüyordu.** "registered_coverage
   is registered/dispatched methods" diyordu; öyle bir alan yok, adı
   `registered_methods`. Keşif katmanının kendini tarif ederken yanılması, sıfır
   bilgili çağıranın ilk okuduğu cümlenin yanlış olması demek.

2. ★★★ **`lights.set_param` ve `material.set` genel `param`/`value`
   setter'ları** ve geçerli anahtarları **şemada hiç görünmüyordu**. Ajan
   `spot_falloff` veya `base_color` demeyi tahmin etmek zorundaydı. 30 malzeme
   ve 6 ışık anahtarı dispatch'ten çıkarılıp enum olarak ilan edildi. Denetim
   bir *belge* hatası arıyordu, bir *keşfedilebilirlik* deliği buldu.

3. ★★★ **Yetki aynası kaymıştı.** `scripts/audit_ipc_capabilities.py`'deki
   `required()` fonksiyonu `RtIpcSecurity.cpp::requiredCapabilities()`'in **elle
   yazılmış kopyası**. C++ altı `sim_graph` okuyucusu sayarken ayna üçünü
   sayıyordu, ve `render.start`'ın `FilesWrite` gereksinimini hiç bilmiyordu.
   Sonuç: **descriptor tablosu yetkiler konusunda yalan söylüyordu** —
   `sim_graph.couplings` için `SceneWrite` diyordu, motor `Read` ile kabul
   ediyor; `render.start` için `Render` diyordu, motor `Render|FilesWrite`
   istiyor.

   ★ Bunu kimse fark etmemişti çünkü denetim **dispatch ile aynayı**
   karşılaştırıyordu, aynayı **kopyaladığı kodla** değil. Artık
   `mirror_drift()` C++'ın açıkça isimlendirdiği 41 metodu ayrıştırıyor ve
   aynanın onaylamasını şart koşuyor.

## Bu denetimin ölçemediği

Dürüst sınır: **davranışsal** iddialar. "Gönderilmeyen alanlar değerini korur"
ya da "bake'i geçersiz kılar" cümleleri bu yöntemle doğrulanamaz. 311 metotta
yalnızca **44 mekanik iddia** bulundu — yani descriptor nesrinin büyük kısmı
hâlâ doğrulanmamış durumda, ve bunu bilmek metriğe güvenmenin şartı.

★ Sıradaki adım bu boşluk için: **getter/setter gidiş-dönüş denetimi.** Her
`X.set_*` için eşleşen okuyucuyu çağırıp "yazdığın değer geri geliyor mu" diye
sormak, bu deponun en pahalı hata sınıfını (sessiz no-op:
`set_transform` scale'i yutuyordu, gaz shader preset'i yok sayılıyordu) genel
olarak hedefler. Henüz yazılmadı.
