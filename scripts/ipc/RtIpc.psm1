# RtIpc.psm1 — RayTrophi Studio'nun yerel IPC pipe'ına PowerShell istemcisi.
#
# Amac: ajanlarin (ve senin) calisan uygulamayi disaridan surmesi. Test yuku
# tek kisiye binmesin diye kurulan IPC katmaninin istemci tarafi.
#
# ── NEDEN TOKEN YOK ─────────────────────────────────────────────────────────
# Yerel tasima `\\.\pipe\RayTrophiStudio`, ve RtIpc.cpp::processJsonMessage
# kimlik dogrulamasini YALNIZCA `context.remote` icin yapiyor. Pipe'in guvenlik
# siniri ACL: RtIpcTransportLocal.cpp::makeSecurity yalnizca SYSTEM +
# Administrators + OLUSTURAN KULLANICININ SID'i icin erisim veriyor. Yani
# token eklemek guvenlik katmiyor, sadece iki yerde tutulacak bir sir yaratirdi.
#
# Token ve RAYTROPHI_REMOTE_IPC_* ortam degiskenleri UZAK (TLS) tasima icindir;
# Start-RayTrophi.ps1 -Remote onu kurar.
#
# ★ Pipe TEK ISTEMCI kabul ediyor (CreateNamedPipeW nMaxInstances = 1). Isin
# bitince Disconnect-RtIpc cagir, yoksa bir sonraki oturum baglanamaz.

$script:RtPipe = $null
$script:RtNextId = 1

function Connect-RtIpc {
    <#
    .SYNOPSIS
        Calisan RayTrophi Studio'ya baglanir.
    .PARAMETER TimeoutMs
        Pipe hazir degilse ne kadar beklenecegi. Uygulama acilirken IPC sunucusu
        Python baslatildiktan SONRA ayaga kalkiyor, o yuzden acilistan hemen
        sonra birkac saniye beklemek normal.
    #>
    [CmdletBinding()]
    param([int]$TimeoutMs = 15000)

    if ($null -ne $script:RtPipe -and $script:RtPipe.IsConnected) {
        Write-Verbose "Zaten bagli."
        return
    }

    $pipe = New-Object System.IO.Pipes.NamedPipeClientStream(
        '.', 'RayTrophiStudio',
        [System.IO.Pipes.PipeDirection]::InOut,
        [System.IO.Pipes.PipeOptions]::None,
        [System.Security.Principal.TokenImpersonationLevel]::Impersonation)
    try {
        $pipe.Connect($TimeoutMs)
    } catch {
        $pipe.Dispose()
        throw "RayTrophi Studio'nun IPC pipe'ina baglanilamadi. Uygulama acik mi, ve SceneLog'da 'IPC server started' satiri var mi? ($($_.Exception.Message))"
    }
    # Sunucu PIPE_TYPE_MESSAGE ile acildi: her yazma bir istek, her mesaj bir
    # yanit. Byte moduna dusersek yanitin nerede bittigini bilemeyiz.
    $pipe.ReadMode = [System.IO.Pipes.PipeTransmissionMode]::Message
    $script:RtPipe = $pipe
    $script:RtNextId = 1
}

function Disconnect-RtIpc {
    [CmdletBinding()]
    param()
    if ($null -ne $script:RtPipe) {
        try { $script:RtPipe.Dispose() } catch {}
        $script:RtPipe = $null
    }
}

function Invoke-RtIpc {
    <#
    .SYNOPSIS
        Tek bir IPC metodu cagirir ve sonucu PowerShell nesnesi olarak dondurur.
    .EXAMPLE
        Invoke-RtIpc version
        Invoke-RtIpc physics.fracture_group @{ group = 'Beam' }
        Invoke-RtIpc project.save @{ path = 'C:\tmp\t.rtp' }
    .PARAMETER Raw
        Hatalari firlatmak yerine ham yaniti dondurur. Bir metodun BASARISIZ
        olmasini bekleyen testler icin.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true, Position = 0)][string]$Method,
        [Parameter(Position = 1)][hashtable]$Params = @{},
        [switch]$Raw
    )

    if ($null -eq $script:RtPipe -or -not $script:RtPipe.IsConnected) {
        Connect-RtIpc
    }

    $request = @{ id = $script:RtNextId; method = $Method; params = $Params }
    $script:RtNextId++
    # ★ -Depth SART. PowerShell 5.1'de varsayilan 2, ve ic ice params (orn.
    # nodes.apply) sessizce "System.Collections.Hashtable" stringine cevrilir —
    # istek gecerli JSON olur, sadece YANLIS olur.
    $json = $request | ConvertTo-Json -Depth 20 -Compress

    $bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
    $script:RtPipe.Write($bytes, 0, $bytes.Length)
    $script:RtPipe.Flush()

    $buffer = New-Object byte[] 65536
    $memory = New-Object System.IO.MemoryStream
    do {
        $read = $script:RtPipe.Read($buffer, 0, $buffer.Length)
        if ($read -le 0) { throw "IPC baglantisi yanit sirasinda kapandi ('$Method')." }
        $memory.Write($buffer, 0, $read)
    } until ($script:RtPipe.IsMessageComplete)

    $text = [System.Text.Encoding]::UTF8.GetString($memory.ToArray())
    $memory.Dispose()
    $response = $text | ConvertFrom-Json

    if ($Raw) { return $response }
    if ($null -ne $response.PSObject.Properties['error'] -and $null -ne $response.error) {
        throw "IPC '$Method' basarisiz: $($response.error)"
    }
    return $response.result
}

function Test-RtIpcReady {
    <#
    .SYNOPSIS
        Pipe var mi? Baglanmadan bakar, bu yuzden tek-istemci yuvasini tuketmez.
    #>
    [CmdletBinding()]
    param()
    return (Test-Path '\\.\pipe\RayTrophiStudio')
}

function Wait-RtIpcReady {
    <#
    .SYNOPSIS
        Pipe belirene kadar bekler. Uygulama acilirken kullanilir.
    #>
    [CmdletBinding()]
    param([int]$TimeoutSeconds = 120)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-RtIpcReady) { return $true }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

Export-ModuleMember -Function Connect-RtIpc, Disconnect-RtIpc, Invoke-RtIpc,
                              Test-RtIpcReady, Wait-RtIpcReady
