$out = "C:\Users\maxkemal\AppData\Local\Temp\claude\e--RayTrophi-projesi-raytracing-Proje-Moduler\95cf38d5-25fd-49ad-b153-fa10327d63cc\scratchpad"
Import-Module E:\RayTrophi_projesi\raytracing_Proje_Moduler\scripts\ipc\RtIpc.psm1 -Force

# Kamerayi SABITLE: iki kol arasinda tek fark uretici olmali.
$cam = Invoke-RtIpc camera.get
$pos = @([double]$cam.position[0], [double]$cam.position[1], [double]$cam.position[2])
$tgt = @([double]$cam.target[0], [double]$cam.target[1], [double]$cam.target[2])
"kamera sabit: pos={0:N1},{1:N1},{2:N1}  hedef={3:N1},{4:N1},{5:N1}" -f $pos[0],$pos[1],$pos[2],$tgt[0],$tgt[1],$tgt[2]

function Drive([int]$n) {
    for ($i = 0; $i -lt $n; $i++) {
        Invoke-RtIpc camera.set_position @{ position = @(($pos[0] + 0.01 * ($i % 4)), $pos[1], $pos[2]) } | Out-Null
    }
    Invoke-RtIpc camera.set_position @{ position = $pos } | Out-Null
    Invoke-RtIpc camera.set_target   @{ target   = $tgt } | Out-Null
    Start-Sleep -Milliseconds 1200
}

function Shoot([string]$name) {
    Invoke-RtIpc viewport.capture @{ enabled = $true } | Out-Null
    Drive 12
    $s = Invoke-RtIpc viewport.get_screenshot
    [IO.File]::WriteAllBytes("$out\$name.jpg", [Convert]::FromBase64String($s.image_base64))
    Invoke-RtIpc viewport.capture @{ enabled = $false } | Out-Null
    $t = Invoke-RtIpc viewport.frame_timings
    $p = Invoke-RtIpc rayfusion.probe_field
    "{0,-12} kare={1,6:N1} ms  main={2,6:N1}  producer={3,-9} valid={4}/{5} hit={6:N3} shaded={7} trace={8:N2} ms" -f `
        $name, $t.frame_gpu_mean_ms, ($t.stages | Where-Object name -eq 'main_pass').gpu_mean_ms,
        $p.producer, $p.valid, $p.total, $p.hit_fraction, $p.bounce_shaded_hits, $p.trace_ms
}

# A kolu: izlenen uretici + tek diffuse bounce (su an acik)
Drive 30
Shoot 'gi_traced'

# B kolu: gokyuzu bake'ine geri don -- alanin BOYUTU ayni kalir, degisen tek sey URETICI
Invoke-RtIpc rayfusion.set_probe_producer @{ traced = $false } | Out-Null
Drive 40
Shoot 'gi_skybake'

# A kolunu geri kur ve TEKRAR cek: sapma bu ucuncu kolla fiyatlanir
Invoke-RtIpc rayfusion.set_probe_producer @{ traced = $true } | Out-Null
Drive 40
Shoot 'gi_traced2'
