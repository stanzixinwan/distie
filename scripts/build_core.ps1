# Build DistIE C++ core (BlockPool) and run native tests.
# Requires: Visual Studio C++ tools, CMake, pybind11 (`pip install pybind11`).

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$CoreDir = Join-Path $RepoRoot "core"
$BuildDir = Join-Path $CoreDir "build"
$CMake = "C:\Program Files\CMake\bin\cmake.exe"
if (-not (Test-Path $CMake)) {
    $cmakeCmd = Get-Command cmake -ErrorAction SilentlyContinue
    if (-not $cmakeCmd) { throw "cmake not found" }
    $CMake = $cmakeCmd.Source
}

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found" }
$vsPath = & $vswhere -latest -products * -property installationPath
if (-not $vsPath) { throw "Visual Studio with C++ tools not found" }
$vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"

$Python = (Get-Command python).Source
& $Python -c "import pybind11" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "pybind11 missing. Run: python -m pip install pybind11"
}

$configure = "`"$CMake`" -S `"$CoreDir`" -B `"$BuildDir`" -DPython_EXECUTABLE=`"$Python`""
$build = "`"$CMake`" --build `"$BuildDir`" --config Release"
$CTest = Join-Path (Split-Path $CMake) "ctest.exe"
$test = "`"$CTest`" --test-dir `"$BuildDir`" -C Release --output-on-failure"

cmd /c "`"$vcvars`" && $configure && $build && $test"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "C++ core build OK. Python tests:"
& $Python (Join-Path $CoreDir "tests\test_block_pool.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
