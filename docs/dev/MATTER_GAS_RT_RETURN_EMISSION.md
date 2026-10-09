# Matter gas: RT dönüşünde önizleme emission kaybı

2026-10-09: İlk Solid/RayFusion doğru, RT doğru; dönüşte yalnız gri/beyaz
duman. Kullanıcı Blackbody/Color Ramp seçimlerinin korunduğunu doğruladı.

Hatalı Solid durumunda dış IPC okuması: render ve viewport aynı `vdb_id=651`,
aktif NanoVDB (`volume_type=2`), ikisinde de yoğunluk ve sıcaklık adresi mevcut.
`gas.get_shader` fire preset ve blackbody intensity 23.899 bildiriyor.
Sıcaklık adresinin tamamen kaybı bu örnekte elendi. Adresin bulunması doğru
veri veya doğru shader bağlamasını kanıtlamaz. Kök neden henüz doğrulanmadı.
Preset shader alias açıklaması incelemede doğrulanmadı; buna göre değişiklik yok.

Mevcut `render.volume_slots` ortak core/Python/IPC çıktısına gerçek backend
tablosundan emission_mode, color_ramp_enabled, blackbody_intensity,
temperature_scale, temperature_min/max eklendi. GPU struct/SPV ABI değişmedi.
Bu değişiklik teşhis içindir; render hatasının düzeltildiği iddia edilmiyor.
Build/uygulama başlatma yapılmadı. Ortak build bekletme kuralı korunuyor.

Sonraki kullanıcı C++ build'inden sonra, hatalı sahne durmuşken:

```powershell
python scripts/test/rt_read_volume_emission_ipc.py --tag solid_after_rt
```

Okuyucu sahne/mod değişmez. İki backend'in aynı vdb_id için emission mode,
ramp, intensity ve aralığını karşılaştır. Paket farklıysa publication yolu;
aynıysa preview binding, sıcaklık içeriği ve fragment evaluation incelenecek.
İlk doğru önizleme ve RT dönüşündeki çıktıları ayrıca sakla; JSON son okumada
yenilenir. Kaynak script Release kopyasıyla aynı tutulur.
