# RayTrophi Stats Auto-Updater
# Usage: powershell -File scripts/update_readme_stats.ps1

$rootPath = Resolve-Path ".."
$readmePath = Join-Path $rootPath "README.md"
$readmeTrPath = Join-Path $rootPath "README_TR.md"

$extensions = @(".cpp", ".h", ".hpp", ".c", ".cu", ".cuh", ".glsl", ".rgen", ".rmiss", ".rchit", ".rahit", ".rint", ".comp");
$excludeDirs = @("external", "libs", "vcpkg", "build", "x64", ".vs", ".git", "_Unused");
$excludeFiles = @("simdjson.cpp", "simdjson.h", "json.hpp", "stb_image.h", "stb_image_write.h", "tinyexr.h", "ImGuizmo.h", "ImGuizmo.cpp", "PNanoVDB.h");

Write-Host "Counting lines of code..."
$files = Get-ChildItem -Path $rootPath -Recurse | Where-Object {
    $item = $_;
    if ($item.PSIsContainer) { return $false };
    if ($extensions -notcontains $item.Extension.ToLower()) { return $false };
    $fullName = $item.FullName;
    foreach ($dir in $excludeDirs) { if ($fullName -like "*\$dir\*") { return $false } };
    foreach ($f in $excludeFiles) { if ($item.Name -eq $f) { return $false } };
    return $true
}

$totalLines = 0;
foreach ($f in $files) {
    $totalLines += (Get-Content $f.FullName -ErrorAction SilentlyContinue | Measure-Object -Line).Lines;
}

Write-Host "Counting UI elements..."
$uiFiles = Get-ChildItem -Path (Join-Path $rootPath "raytrac_sdl2\source") -Recurse -Include *.cpp, *.h, *.hpp
$uiPatterns = @("ImGui::Drag", "ImGui::Button", "ImGui::Checkbox", "ImGui::Slider", "ImGui::Combo", "ImGui::Selectable", "ImGui::MenuItem", "ImGui::ColorEdit")
$totalUI = 0
foreach ($p in $uiPatterns) {
    $count = ($uiFiles | Select-String -Pattern [regex]::Escape($p) -AllMatches).Matches.Count
    $totalUI += $count
}

$date = Get-Date -Format "yyyy-MM-dd"
$statsBlock = @"
<!-- STATS_START -->
| Metric | Value |
| :--- | :--- |
| **Files (Source)** | $($files.Count) |
| **Lines of Code** | $($totalLines.ToString("N0")) |
| **UI Control Points** | $($totalUI)+ |
| **Last Updated** | $date |
<!-- STATS_END -->
"@

function Update-File($path, $block) {
    if (Test-Path $path) {
        $content = Get-Content $path -Raw
        if ($content -match "<!-- STATS_START -->[\s\S]*<!-- STATS_END -->") {
            $newContent = $content -replace "<!-- STATS_START -->[\s\S]*<!-- STATS_END -->", $block
            $newContent | Set-Content $path -NoNewline
            Write-Host "Updated $path"
        } else {
            Write-Warning "Placeholder not found in $path"
        }
    }
}

Update-File $readmePath $statsBlock
Update-File $readmeTrPath $statsBlock
