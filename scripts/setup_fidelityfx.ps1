param(
    [string]$ArchivePath,
    [switch]$Download
)

# Prepare dependencies only. This script never builds shaders/the SDK or loads a DLL.
$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$vendorRoot = Join-Path $repoRoot 'RayTrophiStudio\third_party\FidelityFX\1.1.4'
$manifest = Get-Content -LiteralPath (Join-Path $vendorRoot 'manifest.json') -Raw |
    ConvertFrom-Json

if (!$ArchivePath) {
    $ArchivePath = Join-Path $repoRoot 'x64\.sdk-cache\FidelityFX-SDK-v1.1.4.zip'
}
$ArchivePath = [IO.Path]::GetFullPath($ArchivePath)

if (!(Test-Path -LiteralPath $ArchivePath -PathType Leaf)) {
    if (!$Download) {
        throw 'SDK archive missing. Rerun with -Download, or supply -ArchivePath.'
    }
    [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($ArchivePath)) | Out-Null
    $ProgressPreference = 'SilentlyContinue'
    Invoke-WebRequest -Uri $manifest.archive.url -OutFile $ArchivePath
}

if ((Get-Item -LiteralPath $ArchivePath).Length -ne $manifest.archive.size_bytes) {
    throw 'SDK archive size does not match the pinned v1.1.4 package.'
}
if ((Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256).Hash -ne
    $manifest.archive.sha256) {
    throw 'SDK archive SHA-256 mismatch. Nothing was extracted or deployed.'
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [IO.Compression.ZipFile]::OpenRead($ArchivePath)
try {
    foreach ($file in $manifest.files) {
        # Both paths come from the checked-in manifest, never from ZIP enumeration.
        # Unrelated sample executables, DX12 files and import libs are not extracted.
        $entry = $archive.GetEntry($file.archive_path)
        if (!$entry) {
            throw "Missing SDK entry: $($file.archive_path)"
        }
        $destination = [IO.Path]::GetFullPath((Join-Path $vendorRoot $file.local_path))
        if (!$destination.StartsWith($vendorRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
            throw 'Manifest destination escapes the vendor directory.'
        }
        [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($destination)) | Out-Null
        $inputStream = $entry.Open()
        try {
            $outputStream = [IO.File]::Create($destination)
            try {
                $inputStream.CopyTo($outputStream)
            } finally {
                $outputStream.Dispose()
            }
        } finally {
            $inputStream.Dispose()
        }
        if ((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash -ne $file.sha256) {
            throw "Extracted SDK file hash mismatch: $($file.local_path)"
        }
    }
} finally {
    $archive.Dispose()
}

$dll = Join-Path $vendorRoot 'runtime\amd_fidelityfx_vk.dll'
$signature = Get-AuthenticodeSignature -LiteralPath $dll
if ($signature.Status -ne 'Valid' -or
    $signature.SignerCertificate.Subject -notmatch 'CN=Advanced Micro Devices(?:,|$)') {
    throw "AMD DLL signature validation failed: $($signature.Status)"
}

$destinations = @((Join-Path $repoRoot 'x64\Release'))
$debugRoot = Join-Path $repoRoot 'x64\Debug'
if (Test-Path -LiteralPath $debugRoot -PathType Container) {
    $destinations += $debugRoot
}
foreach ($directory in $destinations) {
    [IO.Directory]::CreateDirectory($directory) | Out-Null
    $target = Join-Path $directory 'amd_fidelityfx_vk.dll'
    Copy-Item -LiteralPath $dll -Destination $target
    Copy-Item -LiteralPath (Join-Path $vendorRoot 'LICENSE.txt') `
        -Destination (Join-Path $directory 'FidelityFX-LICENSE.txt')
    if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -ne
        $manifest.runtime.sha256) {
        throw "Deployed DLL hash mismatch: $target"
    }
    Write-Output "Verified AMD Vulkan runtime: $target"
}
Write-Output 'FidelityFX SDK 1.1.4 staged. No SDK build or DLL execution performed.'
