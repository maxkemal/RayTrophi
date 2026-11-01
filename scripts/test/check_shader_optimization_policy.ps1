# Non-build tests: extract functions via AST; never execute the build driver,
# glslc, spirv-val, or any real shader compilation/optimization process.
$ErrorActionPreference = 'Stop'
$driver = Join-Path $PSScriptRoot '..\..\RayTrophiStudio\compile_shaders.ps1'
$tokens = $null
$parseErrors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile(
    [IO.Path]::GetFullPath($driver), [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count -ne 0) { throw ($parseErrors | Out-String) }
$defaultMode = $ast.ParamBlock.Parameters | Where-Object {
    $_.Name.VariablePath.UserPath -eq 'VolumeOptimization'
}
if ($defaultMode.DefaultValue.Value -ne 'Off') {
    throw 'Routine builds must not retry the known failing -O experiment'
}
$shader = Get-Content -LiteralPath (Join-Path $PSScriptRoot `
    '..\..\RayTrophiStudio\source\shaders\volume_closesthit.rchit') -Raw
$backend = Get-Content -LiteralPath (Join-Path $PSScriptRoot `
    '..\..\RayTrophiStudio\source\src\Backend\VulkanBackend.cpp') -Raw
if (!$shader.Contains('GL_EXT_scalar_block_layout') -or
    !$backend.Contains('scalarFeatures.scalarBlockLayout = supportedScalarFeatures.scalarBlockLayout;') -or
    !$backend.Contains('*nextLink = &scalarFeatures;')) {
    throw 'Shader scalar layout and runtime device-feature wiring disagree'
}
foreach ($name in @('Publish-Spirv', 'Compile-Shader')) {
    $definition = $ast.FindAll({
        param($node)
        $node -is [Management.Automation.Language.FunctionDefinitionAst]
    }, $true) | Where-Object { $_.Name -eq $name }
    if (@($definition).Count -ne 1) { throw "Missing unique function: $name" }
    Invoke-Expression $definition.Extent.Text
}
$script:originalPublish = (Get-Item Function:Publish-Spirv).ScriptBlock
function Publish-Spirv {
    param([string]$TemporaryPath, [string]$TargetPath)
    $script:publishes++
    if ($script:rejectPublish) { throw 'Mock destination unavailable' }
    & $script:originalPublish $TemporaryPath $TargetPath
}

function Mock-Compiler {
    $callArgs = @($args)
    $script:compilerCalls++
    $optimized = $callArgs -ccontains '-O'
    $script:lastOptimized = $optimized
    $script:flagHistory += ($callArgs -join ' ')
    if ($script:busyCalls -gt 0) {
        $script:busyCalls--
        $global:LASTEXITCODE = 1
        'cannot open output file'
        return
    }
    if ($script:failAll -or ($script:failOptimized -and $optimized)) {
        $global:LASTEXITCODE = -1073741571
        'mock stack overflow'
        return
    }
    $outputIndex = [Array]::IndexOf($callArgs, '-o') + 1
    $bytes = New-Object byte[] 20
    [BitConverter]::GetBytes([uint32]0x07230203).CopyTo($bytes, 0)
    # Distinguish publication of the optimized module from the fallback.
    $bytes[16] = if ($optimized) { 11 } else { 22 }
    [IO.File]::WriteAllBytes($callArgs[$outputIndex], $bytes)
    $global:LASTEXITCODE = 0
}

function Mock-Validator {
    if (!(@($args) -ccontains '--scalar-block-layout')) {
        throw 'Validator must enable the scalar layout used by the shader/device'
    }
    $script:validatorCalls++
    $global:LASTEXITCODE = if ($script:rejectOptimized -and $script:lastOptimized) { 1 } else { 0 }
}

function Require([bool]$Condition, [string]$Message) {
    if (!$Condition) { throw $Message }
}

$compilerPath = 'Mock-Compiler'
$validatorPath = 'Mock-Validator'
# Workspace-local ignored scratch permits atomic File.Replace in the sandbox;
# the redirected Windows temporary directory can reject replacement semantics.
$testParent = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\docs\dev\.captures'))
$testRoot = Join-Path $testParent ('rt-shader-policy-' + [Guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($testRoot) | Out-Null
$source = Join-Path $testRoot 'volume_closesthit.rchit'
$target = Join-Path $testRoot 'volume_closesthit.spv'
[IO.File]::WriteAllText($source, 'mock source; never passed to a shader compiler')
$passed = 0
try {
    foreach ($case in @('optimized', 'stack_fallback', 'validator_fallback', 'off',
            'full_failure', 'full_validation_failure', 'both_fail', 'publish_failure', 'busy_then_fallback',
            'busy_exhausts_retries')) {
        $script:compilerCalls = 0
        $script:validatorCalls = 0
        $script:publishes = 0
        $script:flagHistory = @()
        $script:failOptimized = $case -in @('stack_fallback', 'full_failure', 'busy_then_fallback',
            'busy_exhausts_retries')
        $script:failAll = $case -eq 'both_fail'
        $script:rejectOptimized = $case -in @('validator_fallback', 'full_validation_failure')
        $script:rejectPublish = $case -eq 'publish_failure'
        $script:busyCalls = if ($case -eq 'busy_then_fallback') { 1 } elseif (
            $case -eq 'busy_exhausts_retries') { 3 } else { 0 }
        [IO.File]::WriteAllBytes($target, [byte[]](1, 2, 3, 4))
        $flags = @('--target-env=vulkan1.3', '--target-spv=spv1.4', '-O')
        if ($case -eq 'off') { $flags[-1] = '-O0' }
        $fallback = $case -notin @('full_failure', 'full_validation_failure')
        $failed = $false
        try {
            Compile-Shader $source $target $flags -OptimizationFallback $fallback -ValidateOutput $true
        } catch {
            $failed = $true
            Write-Host ('  Caught: ' + $_.Exception.Message)
        }
        $expectedFailure = $case -in @('full_failure', 'full_validation_failure', 'both_fail', 'publish_failure',
            'busy_exhausts_retries')
        Require ($failed -eq $expectedFailure) "Incorrect outcome: $case"
        $actual = [IO.File]::ReadAllBytes($target)
        if ($expectedFailure) {
            Require (($actual -join ',') -eq '1,2,3,4') "Old SPV replaced on failure: $case"
        } else {
            $expectedMarker = if ($case -eq 'optimized') { 11 } else { 22 }
            Require ($actual[16] -eq $expectedMarker) "Wrong optimization published: $case"
            Require ($script:publishes -eq 1) "Unvalidated output published: $case"
        }
        if ($case -eq 'validator_fallback') {
            Require ($script:validatorCalls -eq 2) 'Fallback was not revalidated'
        }
        if ($case -in @('full_failure', 'full_validation_failure', 'publish_failure')) {
            Require ($script:compilerCalls -eq 1) 'Strict/publication failure incorrectly retried'
        }
        if ($case -eq 'both_fail') {
            Require ($script:compilerCalls -eq 2) 'Fallback must be attempted only once'
            Require ($script:publishes -eq 0) 'Failed output reached publication'
        }
        Require (@(Get-ChildItem -LiteralPath $testRoot -Filter '.shader-*.tmp').Count -eq 0) `
            "Temporary shader leaked: $case"
        $passed++
        Write-Host "PASS $case"
    }
    Write-Host "PASS: $passed mock policy cases; no compiler or build executed"
} finally {
    # Verify the absolute temporary target before recursive cleanup.
    $resolved = [IO.Path]::GetFullPath($testRoot)
    $prefix = $testParent.TrimEnd('\') + '\'
    if (!$resolved.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase) -or
        ![IO.Path]::GetFileName($resolved).StartsWith('rt-shader-policy-')) {
        throw "Unsafe test cleanup path: $resolved"
    }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
