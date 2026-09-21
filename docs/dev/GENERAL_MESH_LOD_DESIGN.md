# Genel mesh LOD — geometriden BAĞIMSIZ tasarım

> **Durum:** TASLAK — tasarım kararı (2026-09-08), kod yazılmadı.
> Gerekçe ve ölçümler: [RASTER_MICROTRIANGLE_WALL.md](RASTER_MICROTRIANGLE_WALL.md)
> §8 (uçurum) ve §7 (maliyet ayrışması).

## 1. Gereksinim: varlığa özel OLMAYACAK

★★★★ İlk taslak iğne/twig üçgenlerini **materyal adından** seçip seyreltiyordu.
Kullanıcı bunu reddetti, ve haklı: o çözüm *bu iki çam ağacına* çalışır,
kullanıcının kendi nesnesine hiçbir şey yapmaz. LOD **genel ve güçlü** olmalı.

★ Ama materyal analizinden çıkan **ölçüm** hâlâ geçerli ve tasarımı yönlendiriyor:
ağacın %84'ü **kopuk, küçük, alfa-test'li kartlar**. Bu bir *materyal* özelliği
değil, **yapısal** bir özellik — ve yapısal özellik genelleştirilebilir.

## 2. Neden tek bir sadeleştirici yetmez

| Girdi | Doğru işlem | Yanlış işlem |
|---|---|---|
| Bağlantılı yüzey (gövde, karakter, bina) | QEM kenar çökertme | kart atma → delik |
| Çok sayıda küçük kopuk parça (yaprak kartı, iğne, çakıl, perçin, zincir halkası) | **parçayı bütün olarak SİL**, kalanları alan telafisiyle büyüt | QEM → 2 üçgenlik kartta ya hiç azalma ya yok olma |

★★★ İki üçgenlik alfa-test'li bir kartta QEM'in **ara adımı yoktur**. Foliage'da
genel QEM'in çöp üretmesinin sebebi budur — eksik olan şey "foliage bilgisi"
değil, **parça yapısına duyarlılık**.

## 3. Tasarım: parça-duyarlı sadeleştirme (component-aware)

Hiçbir adımda "bu bitki mi?" diye sorulmaz. Sorular yapısaldır.

```
flat SoA  ->  [1] weld  ->  [2] baglantili parcalar  ->  [3] siniflandir
                                                            |
                                    buyuk/baglantili --------+------- kucuk & cok sayida
                                            |                              |
                                    [4a] QEM cokertme            [4b] parca silme + alan telafisi
                                            \______________  ______________/
                                                           \/
                                              [5] kademe mesh'i (ucgen hedefi)
```

**[1] Weld.** Flat SoA'da bağlantı yoktur (indeks tamponu yok — bu bir tercih).
Konum bazlı weld ayrı bir önbellek olarak kurulur; depoda örneği var:
`scene_data.h:772` `flat_soa_to_unique` (erime yolu). Geometriye gömülmez.

**[2] Parçalar.** Weld edilmiş bağlantı üzerinde union-find.

**[3] Sınıflandırma** — eşik nesnenin kendi bbox'ına **görelidir**, mutlak
birimde değil; böylece bina da çakıl da doğru sınıflanır. "Küçük parçadan çok
sayıda var" ölçütü hem yaprak kartına hem perçine uyar.

**[4a] QEM.** Nitelik duyarlı: materyal sınırı ve UV dikişi çökertilmez
(çökertilirse doku kayar — bu deponun `texCoord` dersinin aynısı).

**[4b] Parça silme.** Kalanlar, kaybedilen **projeksiyon alanını** telafi edecek
kadar büyütülür. Görünüm istatistiksel olarak korunur; mesafede önemli olan budur.
★ Silme deterministik olmalı (parça merkezinden hash), yoksa kademeler arası
geçişte kaynaşma olur.

**[5] Kademeler** üçgen hedefiyle üretilir, elle değil.

## 4. Seçim ölçütü: MESAFE değil EKRAN HATASI

Bugünkü cull shader mesafe eşiği tutuyor (`lodDistSq`). Genel olması için ölçüt
**piksel cinsinden ekran hatası** olmalı:

```
screenError = worldError / distance * focalPixels
```

Böylece bina ile çakıl aynı kadranla doğru davranır; eşik varlık boyutundan
bağımsızlaşır.

★★★★ Bu, kullanıcının önerdiği cluster boru hattının **"Screen Error
Evaluation"** adımının ta kendisidir. Yani burada yazılan şey ileride atılmaz —
cluster'a geçilirse **ön uç aynen kalır**, değişen yalnızca arkasındaki
temsil (mesh kademesi yerine cluster) olur.

## 5. Yapısal engel: LOD zinciri İKİ kademeye sabitlenmiş

`RasterGpuCull::MeshBinding` tek proxy taşıyor (`proxyDrawSlot`,
`proxyElementCount`, `proxyOutBase`, `proxyFlags`) ve `classify()` bir **bool**
seçiyor. Gereken: kademe dizisi + eşik dizisi, `classify()` bir **indeks**
seçer, mesh başına 2 yerine N draw slot.

★ Bu parça geometriden zaten bağımsız — mevcut 96 üçgenlik billboard son kademe
olarak yerinde kalır.

## 6. Sıra ve ölçüt

| # | İş | Bağımsız değeri |
|---|---|---|
| 1 | LOD zincirini N kademeye aç (§5) | Tek başına ölçülebilir: mevcut proxy'yi 2 eşiğe böl |
| 2 | Weld + parça analizi (§3.1-3.3) | `lod.analyze` ile raporlanabilir: parça sayısı/boyut dağılımı |
| 3 | [4b] parça silme | Foliage'da kazancın çoğu burada (%84) |
| 4 | [4a] QEM | Kullanıcı nesneleri ve gövde/bina için |
| 5 | Ekran hatası ölçütü (§4) | Cluster'a geçişte korunur |

★★ **Her adım IPC'ye açılır** (CLAUDE.md kural 1): `lod.analyze`, `lod.build`,
`lod.get`, `lod.set_thresholds`. Aksi hâlde tek kişilik bakımda test edilemez —
ve LOD tam olarak "sessizce makul görünen" arıza sınıfıdır.

## 7. Kabul ölçütü

- Üçgen: 99 000 → ~48 600 (L1) → ~28 000 (L2). Fusion tahmini 433 → ~210 → ~120 ms
  (§7c'deki 15 ms/Müçgen doğrusallığından; **tahmin**).
- ★★★ Görsel: kademe geçişinde **siluet korunur**. En sinsi başarısızlık,
  üçgen sayısı hedefe otururken siluetin incelmesi — sayı doğru, ağaç seyrek.
- Genellik testi: foliage OLMAYAN bir kullanıcı nesnesinde (karakter, bina)
  kademeler delik açmadan üretilebilmeli.
