# Run Debug algorithm V5 from any working directory; forward all CLI arguments.
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$program = Join-Path $repoRoot 'build/bin/x64/Debug/v5/multi-view-3d-reconstruction.exe'
if (-not (Test-Path -LiteralPath $program)) {
    throw 'Build Debug x64 algorithm V5 first. See docs/building.md.'
}
Push-Location -LiteralPath $repoRoot
try {
    & $program @args
    $programExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $programExitCode
