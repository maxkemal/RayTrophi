<#
.SYNOPSIS
    RayTrophi Studio'yu baslatir ve IPC hazir olana kadar bekler.

.DESCRIPTION
    Build aldiktan sonra tek komut: bu scripti calistir, "HAZIR" yazisini gor,
    ajana devret. Uygulamayi kapatmaz ve hicbir sahne dosyasina dokunmaz.

    ── Yerel mi, uzak mi ────────────────────────────────────────────────────
    VARSAYILAN yerel: `\\.\pipe\RayTrophiStudio`, token YOK. Bu bir eksiklik
    degil — RtIpc.cpp kimlik dogrulamasini yalnizca uzak istekler icin yapiyor,
    yerel pipe'in guvenlik siniri ACL (yalnizca seni, SYSTEM ve Administrators
    kabul ediyor). Token eklemek guvenlik katmaz, iki yerde tutulacak bir sir
    yaratir.

    -Remote ile TLS dinleyicisi de acilir. ISTE O ZAMAN token gerekir (en az 32
    karakter, RtIpc.cpp bunu dogruluyor); script uretir, ekrana BIR KEZ basar ve
    ortam degiskenine koyar. Yalnizca ajan baska bir makinedeyse gerekli.

.EXAMPLE
    .\Start-RayTrophi.ps1
.EXAMPLE
    .\Start-RayTrophi.ps1 -Remote -AllowFiles
#>
[CmdletBinding()]
param(
    [string]$ExePath,
    # Ac ve BEKLEME: uygulama zaten acıksa yenisini baslatma, sadece dogrula.
    [switch]$AttachOnly,
    # Uzak TLS dinleyicisi + bootstrap token. Ajan baska makinedeyse.
    [switch]$Remote,
    # project.open / scene.import_model / script.run_file gibi dosya OKUYAN
    # metotlar icin. Yerelde etkisi yok (yerel zaten yetki kontrolune girmiyor).
    [switch]$AllowFiles,
    # script.run_file. Ayri opt-in, cunku keyfi Python calistirmak demek.
    [switch]$AllowScripts,
    [string]$AllowCidrs = '127.0.0.1/32',
    [int]$TimeoutSeconds = 180
)

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Import-Module (Join-Path $here 'RtIpc.psm1') -Force

if (-not $ExePath) {
    $repo = Split-Path -Parent (Split-Path -Parent $here)
    $ExePath = Join-Path $repo 'x64\Release\RayTrophiStudio.exe'
}

# ── Zaten calisiyor mu ──────────────────────────────────────────────────────
$running = Get-Process -Name 'RayTrophiStudio' -ErrorAction SilentlyContinue
if ($running) {
    Write-Host "RayTrophi Studio zaten calisiyor (PID $($running[0].Id))." -ForegroundColor Yellow
    if (-not $AttachOnly) {
        Write-Host "Yenisini baslatmiyorum. Yeni bir surumu test edecekseniz once kapatin." -ForegroundColor Yellow
    }
} elseif ($AttachOnly) {
    throw "-AttachOnly verildi ama calisan bir RayTrophi Studio yok."
} else {
    if (-not (Test-Path $ExePath)) {
        throw "Calistirilabilir bulunamadi: $ExePath  (build alindi mi?)"
    }

    # ★ Yasin kendisi test edilenden ONEMLI. Eski bir exe ile yeni bir kodu test
    # etmek, bu projede daha once tam bir turu yakti: panelde olmayan bir kutu
    # arandi. Bu yuzden zaman damgasini her seferinde basiyoruz.
    $exe = Get-Item $ExePath
    $ageMinutes = [int]((Get-Date) - $exe.LastWriteTime).TotalMinutes
    Write-Host ("Exe: {0}" -f $exe.FullName)
    Write-Host ("     {0}  ({1} dakika once derlendi)" -f $exe.LastWriteTime, $ageMinutes) -ForegroundColor DarkGray

    if ($Remote) {
        $bytes = New-Object byte[] 32
        [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
        $token = ([System.BitConverter]::ToString($bytes) -replace '-', '').ToLower()

        $env:RAYTROPHI_REMOTE_IPC = '1'
        $env:RAYTROPHI_REMOTE_IPC_TOKEN = $token
        $env:RAYTROPHI_REMOTE_IPC_ALLOW_CIDRS = $AllowCidrs
        if ($AllowFiles)   { $env:RAYTROPHI_REMOTE_IPC_ALLOW_FILES = '1' }
        if ($AllowScripts) { $env:RAYTROPHI_REMOTE_IPC_ALLOW_SCRIPTS = '1' }
        $env:RAYTROPHI_REMOTE_IPC_AUDIT_JSONL = Join-Path $env:TEMP 'raytrophi_ipc_audit.jsonl'

        Write-Host ""
        Write-Host "UZAK IPC TOKEN (bir kez gosterilir, saklanmaz):" -ForegroundColor Cyan
        Write-Host "  $token" -ForegroundColor Cyan
        Write-Host "  izinli CIDR: $AllowCidrs" -ForegroundColor DarkGray
        Write-Host "  denetim: $($env:RAYTROPHI_REMOTE_IPC_AUDIT_JSONL)" -ForegroundColor DarkGray
        Write-Host ""
    }

    Start-Process -FilePath $ExePath -WorkingDirectory (Split-Path -Parent $ExePath) | Out-Null
    Write-Host "Baslatildi, IPC bekleniyor..." -ForegroundColor DarkGray
}

# ── Pipe'in belirmesini bekle ────────────────────────────────────────────────
# IPC sunucusu Python baslatildiktan SONRA ayaga kalkiyor (Main.cpp), yani
# pencere gorunur olduktan sonra da birkac saniye gecebilir.
if (-not (Wait-RtIpcReady -TimeoutSeconds $TimeoutSeconds)) {
    throw "IPC pipe $TimeoutSeconds saniyede belirmedi. SceneLog.txt'de 'IPC server' satirina bakin."
}

# ── Gercekten konusabiliyor muyuz ───────────────────────────────────────────
# Pipe'in VAR olmasi cevap verecegini kanitlamaz; bir tur atmadan "hazir"
# demeyiz. Bu projede "tripwire'in susmasi yokluğu kanitlamaz" dersi pahaliya
# ogrenildi.
try {
    Connect-RtIpc
    $version = Invoke-RtIpc version
    $project = Invoke-RtIpc project.path
} finally {
    Disconnect-RtIpc
}

Write-Host ""
Write-Host "HAZIR" -ForegroundColor Green
Write-Host "  surum : $($version | ConvertTo-Json -Compress -Depth 5)"
Write-Host "  proje : $($project | ConvertTo-Json -Compress -Depth 5)"
Write-Host ""
Write-Host "Ajan artik surebilir. Elle denemek icin:" -ForegroundColor DarkGray
Write-Host "  Import-Module .\scripts\ipc\RtIpc.psm1 -Force" -ForegroundColor DarkGray
Write-Host "  Invoke-RtIpc scene.list" -ForegroundColor DarkGray
