# Internal build helper. Dot-sourcing defines functions; it invokes no tool.
# Production entrypoint disables this driver-rejected route; retained for mock
# staging coverage and diagnosis, not as a supported shader installation path.
function Invoke-VolumeBuildTool {
    param([string]$Tool, [string[]]$ToolArguments, [string]$Stage)
    $savedPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $global:LASTEXITCODE = $null
        $messages = @(& $Tool @ToolArguments 2>&1)
        $toolExit = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $savedPreference
    }
    foreach ($message in $messages) { Write-Host $message.ToString() }
    if ($null -eq $toolExit -or $toolExit -ne 0) {
        throw "$Stage failed (exit=$toolExit); previous SPV preserved"
    }
}

function Compile-ModularVolumeShader {
    param(
        [string]$SourcePath,
        [string]$TargetPath,
        [string[]]$CompilerFlags,
        [string]$PythonPath,
        [string]$OptimizerPath,
        [string]$BoundaryToolPath
    )
    if (!($CompilerFlags -contains '-O0') -or ($CompilerFlags -contains '-O')) {
        throw 'Modular volume compilation requires an unoptimized frontend input.'
    }
    $directory = [IO.Path]::GetDirectoryName($TargetPath)
    $stamp = [guid]::NewGuid().ToString('N')
    $reference = Join-Path $directory ('.shader-modular-' + $stamp + '-reference.tmp')
    $protected = Join-Path $directory ('.shader-modular-' + $stamp + '-boundaries.tmp')
    $optimized = Join-Path $directory ('.shader-modular-' + $stamp + '-optimized.tmp')
    try {
        # The ordinary compiler helper stages/validates this private reference.
        # Its publication target is not the working volume_closesthit.spv.
        Compile-Shader $SourcePath $reference $CompilerFlags -ValidateOutput $true
        Invoke-VolumeBuildTool $PythonPath @(
            $BoundaryToolPath, $reference, '--output', $protected, '--summary') 'Function boundaries'
        Invoke-VolumeBuildTool $validatorPath @(
            '--target-env', 'vulkan1.3', '--scalar-block-layout', $protected) `
            'Function boundary validation'
        Invoke-VolumeBuildTool $OptimizerPath @(
            '--target-env=vulkan1.3', '--scalar-block-layout', '-O',
            $protected, '-o', $optimized) 'Modular optimization'
        Invoke-VolumeBuildTool $validatorPath @(
            '--target-env', 'vulkan1.3', '--scalar-block-layout', $optimized) `
            'Modular output validation'
        # Only the fully optimized, validated result may replace the old module.
        Publish-Spirv -TemporaryPath $optimized -TargetPath $TargetPath
        Write-Host ('  OK: ' + [IO.Path]::GetFileName($TargetPath) +
            ' [Modular: glslc -O0 -> DontInline -> spirv-opt -O; validated]')
    } finally {
        foreach ($temporary in @($reference, $protected, $optimized)) {
            if ([IO.File]::Exists($temporary)) {
                Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
            }
        }
    }
}
