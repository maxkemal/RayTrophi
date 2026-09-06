# Foliage yerlesim maskeleri — sozlesme

> **Durum:** REFERANS — 2026-08-30. Bir tasarim plani degil, baglayici bir
> karar: maske slotlarinin sayisi ve birlesimin NEREDE yapildigi.

## Iki rol, ve simetrik DEGILLER

Katman basina iki alan slotu var ve ikisi ayni sey degildir:

| Slot | Semantik | Shader |
|---|---|---|
| `density_mask` | **DAHIL, olasiliksal** — deger bir yerlesim OLASILIGIDIR | `random01 > density ⇒ red` |
| `exclusion_mask` | **DISLA, sert esik** — yasak bolge | `deger >= exclusion_threshold ⇒ red` |

0.3'luk bir yogunluk bir mesceyi **seyreltir**; 0.3'luk bir dislama alani hicbir
sey yapmaz (esigin altinda). Bu ayrim kasitlidir ve korunmalidir. Ayrica splat
haritasindan bir dahil ve bir disla kanali var; `exclusion_threshold` **hem**
alani **hem** splat kanalini surer (tek kadran, iki tuketici).

## KARAR: slot sayisi artmaz

"Gol + nehir + kayalik" gibi coklu dislama istegi gercektir, ama cevabi
tuketiciye N slot eklemek **degildir**:

1. **Birlesim graph'in isidir.** Maske bir ALAN. Iki alani birlestirmek Math
   node'unun isi. N slot koyarsan birlesim semantigini de (and/or/agirlik/slot
   basina esik) panele koymak zorundasin — panelin icine gizlenmis kucuk bir
   ifade dili, ve o dil zaten var.
2. **Port sadelestirmesi yonune ters.** Bkz.
   [TERRAIN_NODE_CONTRACT_REDESIGN.md](TERRAIN_NODE_CONTRACT_REDESIGN.md);
   budanan/eklenen her pin tuketicide sessiz sifir riskidir.
3. **Ol​cum ile karar ayri kalmali.** Bir maskenin degeri bir olcumdur; o
   olcumlerin bir yerlesim kararina nasil donustugu **yazarlik**tir. Ikisini ayni
   yere koymak, `feedback_measurement_is_not_visibility` dersinin tekrari olur.

## Bunun yerine: Publish Field

`TerrainV2.PublishField` (2026-08-30) bir alani **yazarin sectigi bir isimle**
yayinlar. Terrain Fields Output'un 36 pini sabittir cunku her biri BELIRLI bir
olculen buyukluktur; bestelenmis bir maske ise bir olcum degil bir karardir ve
kendi ismini hak eder.

Akis: Math node'lariyla birlestir → `Publish Field` ile `mask.no_water` diye
yayinla → tek maske slotunu ona yonelt.

Kurallar:
- Ad normalize edilir (kucuk harf, gecersiz karakterler `_`, namespace yoksa
  `mask.` on eki).
- **Olculen 36 isim rezervedir**; uzerine yazmak reddedilir. Ayni isim ≠ ayni
  is: bir tuketicinin `terrain.slope` okuyup baska bir sey almasi sessiz bir
  ariza olurdu.
- Boyut uyusmazligi reddedilir. Scatter shader'i yanlis boyutlu alan icin notr
  fallback'e duser, yani "maske kapali" gibi gorunur — hata gibi degil.
- Panelde ne yayinlandigi / neden yayinlanmadigi **yaziyor**. Sessizce hicbir sey
  yapmayan bir yayinci, bu node'un davet edecegi ariza olurdu.

## Isim tuketicileri CANLI listeden beslenir

Foliage layer alan seciciler artik terrain'in **gercekten yayinladigi**
`analysisFields` anahtarlarini listeler (sabit 13 isim degil). Sabit liste iki
yonde birden yaniltiyordu: terrain'de olmayan isimleri oneriyordu (maske notr
fallback'e duser, sessizce hicbir sey yapmaz) ve `Publish Field`'in urettigi
ismi **hic** gosteremiyordu. Yayinlanmamis bir ad secili kalirsa silinmez ama
**isaretlenir**.

Script tarafi: `terrain.list_fields` ayni olcumu dondurur.

## Script yuzeyi

- `scatter.set_settings` — dahil/disla/olcek maskeleri, esik, splat kanallari.
  Yalnizca gecilen anahtarlar yazilir. Aralik disi splat kanali **reddedilir**,
  kirpilmaz (kirpma bir yazim hatasini alpha kanalina maskeleme yapan calisir
  gorunumlu bir ayara cevirirdi).
- `scatter.list_groups` — ayni alanlari **geri okur**. Geri okumasi olmayan bir
  setter test edilemez: yazdigi ad var olan bir alana cozulmese de basari doner.
- `Publish Field`'in `fieldName`'i generic `nodes.set_property` ile yazilir.

Olcum: `scripts/probe_foliage_mask_surface.py`
