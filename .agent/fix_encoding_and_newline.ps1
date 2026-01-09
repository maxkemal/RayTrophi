$sourceDir = "..\raytrac_sdl2" # Relative path to source
$extensions = @("*.c", "*.cpp", "*.h", "*.hpp", "*.cu", "*.cuh")

# Get all source files recursively
$files = Get-ChildItem -Path $sourceDir -Recurse -Include $extensions

foreach ($file in $files) {
    try {
        # Read file content (auto-detects current encoding)
        $content = Get-Content -Path $file.FullName -Raw

        # Check for empty files to avoid errors
        if ($null -eq $content) { continue }


        # 1. Fix End of File (EOF)
        # Check if the file ends with a newline
        if (-not $content.EndsWith("`n")) {
            Write-Host "Fixing EOF newline: $($file.Name)"
            $content += "`r`n"
        }

        # 2. Force UTF-8 with BOM (Visual Studio prefers BOM for C++)
        # We re-save even if content didn't change, just to enforce encoding
        # BUT to avoid touching every file timestamp unnecessarily, we can be smarter.
        # However, user wants to fix encoding, so saving over is the surest way.
        
        # To strictly reset encoding, we always save.
        
        # In newer PowerShell (Core), 'utf8' is no bom. 'utf8BOM' is explicit.
        # In Windows PowerShell 5.1, 'UTF8' implies BOM.
        # determining environment... assumes Windows generally.
        
        # We will use .NET directly to be safe and consistent.
        $utf8WithBom = New-Object System.Text.UTF8Encoding($true)
        
        [System.IO.File]::WriteAllText($file.FullName, $content, $utf8WithBom)
        
        # Write-Host "Processed: $($file.Name)"
    }
    catch {
        Write-Error "Failed to process $($file.Name): $_"
    }
}

Write-Host "Islem tamamlandi. Tum dosyalar UTF-8 (BOM) yapildi ve dosya sonu satiri eklendi."
