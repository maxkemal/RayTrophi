<#
.SYNOPSIS
    RayTrophi Studio'ya tek satirlik IPC komutu. Ajanin ve senin ortak arayuzun.

.DESCRIPTION
    Amac iki tarafli:
      - IZIN: her cagri farkli bir PowerShell metni olunca izin sistemi her
        seferinde soruyor. Tek giris noktasi = tek izin kurali.
      - OKUNABILIRLIK: ekran kaydinda 300 karakterlik tek satir yerine
        `rt terrain.erode @{...}` gorunur.

.EXAMPLE
    .\rt.ps1 version
.EXAMPLE
    .\rt.ps1 terrain.create -P @{ name = 'Ana'; size = 2048 }
.EXAMPLE
    .\rt.ps1 -File scripts\material_phase_tests\phase_10_fracture_roundtrip.py
.EXAMPLE
    .\rt.ps1 -Methods terrain      # terrain.* metotlarini listeler
#>
[CmdletBinding(DefaultParameterSetName = 'Call')]
param(
    [Parameter(ParameterSetName = 'Call', Position = 0)][string]$Method,
    [Parameter(ParameterSetName = 'Call', Position = 1)][Alias('P')][hashtable]$Params = @{},
    # Python dosyasi calistir (script.run_file). Yol goreli olabilir.
    [Parameter(ParameterSetName = 'File')][string]$File,
    # Yerel metot listesini filtreleyip yazdirir; uygulamaya baglanmaz.
    [Parameter(ParameterSetName = 'Methods')][string]$Methods,
    # Ham yanit (hata firlatmaz). Basarisiz olmasi BEKLENEN cagrilar icin.
    [switch]$Raw,
    # Sonucu JSON yerine nesne olarak dondur (boru hattina sokmak icin).
    [switch]$AsObject
)

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent (Split-Path -Parent $here)

if ($PSCmdlet.ParameterSetName -eq 'Methods') {
    # Dispatch tablosundan okunur, elle tutulan bir listeden DEGIL — elle tutulan
    # liste kacinilmaz olarak koddan ayrisir ve olmayan metot onerir.
    $src = Join-Path $repo 'RayTrophiStudio\source\src\Api'
    $all = Select-String -Path (Join-Path $src 'RtIpc*.cpp') `
                         -Pattern 'method == "([a-z_]+\.[a-z_0-9]+)"' -AllMatches |
           ForEach-Object { $_.Matches } | ForEach-Object { $_.Groups[1].Value } |
           Sort-Object -Unique
    if ($Methods) { $all = $all | Where-Object { $_ -like "*$Methods*" } }
    $all
    return
}

Import-Module (Join-Path $here 'RtIpc.psm1') -Force

if ($PSCmdlet.ParameterSetName -eq 'File') {
    # ★ Mutlak yola cevrilir. script.run_file uygulamanin CALISMA DIZININE gore
    # cozer, senin kabuguna gore degil — goreli yol sessizce baska bir dosyayi
    # (ya da hicbir seyi) bulur.
    if (-not [System.IO.Path]::IsPathRooted($File)) {
        $File = Join-Path $repo $File
    }
    if (-not (Test-Path $File)) { throw "Script bulunamadi: $File" }
    $Method = 'script.run_file'
    $Params = @{ path = (Resolve-Path $File).Path }
}

if (-not $Method) { throw "Metot adi gerekli. Ornek: .\rt.ps1 version" }

try {
    $result = Invoke-RtIpc $Method $Params -Raw:$Raw
} finally {
    Disconnect-RtIpc   # pipe TEK istemci kabul ediyor; birakmazsak sonraki cagri asilir
}

if ($AsObject) { $result } else { $result | ConvertTo-Json -Depth 12 }
