$ErrorActionPreference = "Stop"

function Ensure-BinDir {
    $bin = Join-Path $env:USERPROFILE ".local\\bin"
    if (-not (Test-Path $bin)) {
        New-Item -ItemType Directory -Path $bin | Out-Null
    }
    return (Resolve-Path $bin).Path
}

function Ensure-PathContains([string]$dir) {
    $userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
    $parts = @()
    if ($userPath) { $parts = $userPath.Split(";") | Where-Object { $_ -and $_.Trim() } }
    if ($parts -contains $dir) {
        return
    }
    $newPath = if ($userPath) { "$userPath;$dir" } else { $dir }
    [Environment]::SetEnvironmentVariable("PATH", $newPath, "User")
    $env:PATH = "$env:PATH;$dir"
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$bin = Ensure-BinDir

$shim = Join-Path $bin "Sci-Agent-STM.cmd"
$target = Join-Path $repoRoot "Sci-Agent-STM.cmd"

if (-not (Test-Path $target)) {
    throw "Expected $target to exist. Run this from the repo after pulling the bootstrap scripts."
}

Copy-Item -LiteralPath $target -Destination $shim -Force
Ensure-PathContains -dir $bin

Write-Host "Installed: $shim"
Write-Host "Open a new terminal and run: Sci-Agent-STM"
