# NEXT BUILD CHECKS

> **Durum:** CANLI — 2026-10-22; uzun driver compile ölçüldü, geri alma yapılmadı.

1. Bu parti kod değiştirmedi, build gerektirmez. Tamamlanan pipeline495.285sn;
   önceki doğru-sphere build313.558sn. Driver stage feedback bütün stage'lerde
   null: toplamı kesin volume shader süresi diye raporlama.
2. SPIR-V kanıtı rt_phase_cache_spirv_20261022.json: volume4.65MB/252254komut,
   tekfunction,674loop/7585selection/16145phi. Static ölçüm compile süresi değildir.
3. Runtime rt_gas_cost_phasecache_20261022.json: authored185/188ms, ikinci
   scatter0223ms, scatter1463ms, .001485ms, shadow0337ms, stride16398ms,
   density0159ms. UI/scheduler dahil doğal RT wall; saf GPU timestamp değil.
   Ayarlar geri yüklendi, frame101; shader/pipeline kaynakları aynı bırakıldı.
4. Sonraki kontrollü deney yalnız phase-cache var/yok iki kullanıcı build'i.
   İki varyantın SPIR-V hash/boyutunu ve driver total/stage sürelerini sakla.
   Kazanç yoksa veya compile maliyeti artıyorsa kontrollü olarak karar ver;
   sırf source küçük diye güvenli/ucuz sayma. Mevcut çalışan SPV silinmesin.
5. Aynı shader/binary yeniden açılışında warm cache süresi ayrı ölçülsün;
   shader değişimiyle invalid olan ilk derleme warm-cache ile kıyaslanmasın.
6. Performans kararı sonrası görünürlük, gas/splat sıralaması ve düşük katkı
   render kalitesi yeniden kabul edilsin. Fog/emitter blackbody ayrı sonraki iş.

Doğrulama: JSON parse, SPIR-V header/opcode yürüyüşü ve live restore/frame
kontrolleri geçti. Agent build/shader compile veya app launch çalıştırmadı.
