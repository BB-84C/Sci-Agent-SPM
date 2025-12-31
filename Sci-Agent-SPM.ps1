$ErrorActionPreference = "Stop"

function Resolve-RepoRoot {
    $here = $PSScriptRoot
    if (-not $here) {
        $here = Split-Path -Parent $MyInvocation.MyCommand.Path
    }
    if (-not $here) {
        throw "Cannot determine repo root."
    }
    return (Resolve-Path $here).Path
}

function Ensure-Python {
    $py = Get-Command python -ErrorAction SilentlyContinue
    if (-not $py) {
        throw "Python not found on PATH. Install Python 3.11+ and restart your terminal."
    }
}

function Ensure-Venv([string]$repoRoot) {
    $venvDir = Join-Path $repoRoot ".venv"
    $venvPy = Join-Path $venvDir "Scripts\\python.exe"
    if (Test-Path $venvPy) {
        return $venvPy
    }

    Write-Host "Creating venv at $venvDir ..."
    python -m venv $venvDir
    if (-not (Test-Path $venvPy)) {
        throw "Failed to create venv at $venvDir"
    }
    return $venvPy
}

function Ensure-Dependencies([string]$venvPy, [string]$repoRoot) {
    # In some PowerShell versions/configs, non-zero exit codes from native commands can be treated as terminating
    # errors when $ErrorActionPreference="Stop". We want missing deps to fall through to pip install.
    $depsOk = $false
    try {
        & $venvPy -c "import mss, pyautogui, PIL, pynput, textual; from google import genai" *> $null
        if ($LASTEXITCODE -eq 0) { $depsOk = $true }
    }
    catch {
        $depsOk = $false
    }

    if ($depsOk) {
        return
    }

    $req = Join-Path $repoRoot "requirements.txt"
    if (-not (Test-Path $req)) {
        throw "requirements.txt not found."
    }
    Write-Host "Installing dependencies from requirements.txt ..."
    & $venvPy -m pip install --upgrade pip
    & $venvPy -m pip install -r $req
}

function Ensure-Workspace([string]$repoRoot) {
    $ws = Join-Path $repoRoot "workspace.json"
    if (Test-Path $ws) {
        return
    }
    $example = Join-Path $repoRoot "workspace.example.json"
    if (Test-Path $example) {
        Copy-Item -LiteralPath $example -Destination $ws
        Write-Host "Created workspace.json from workspace.example.json. Please calibrate: python -m src.calibrate_gui --workspace workspace.json"
        return
    }
    throw "workspace.json missing. Create one via: python -m src.calibrate_gui --workspace workspace.json"
}

function Ensure-GeminiKey([string]$repoRoot) {
    if ($env:GEMINI_API_KEY -or $env:GOOGLE_API_KEY) {
        return
    }

    $envFile = Join-Path $repoRoot ".env"
    if (Test-Path $envFile) {
        foreach ($line in Get-Content -LiteralPath $envFile) {
            $t = ([string]$line).Trim()
            if (-not $t -or $t.StartsWith("#")) { continue }
            if ($t -match "^\s*GEMINI_API_KEY\s*=\s*(.+)\s*$") {
                $env:GEMINI_API_KEY = $Matches[1].Trim().Trim('"').Trim("'")
                break
            }
            if ($t -match "^\s*GOOGLE_API_KEY\s*=\s*(.+)\s*$") {
                $env:GOOGLE_API_KEY = $Matches[1].Trim().Trim('"').Trim("'")
                break
            }
        }
        if ($env:GEMINI_API_KEY -or $env:GOOGLE_API_KEY) {
            return
        }
    }

    Write-Host "Gemini API key not found (env GEMINI_API_KEY / GOOGLE_API_KEY)."
    $key = Read-Host "Paste your Gemini key for this session (won't be saved)"
    if (-not $key) {
        throw "Missing Gemini API key."
    }
    $env:GEMINI_API_KEY = $key
}

$repoRoot = Resolve-RepoRoot
Set-Location $repoRoot

Ensure-Python
$venvPy = Ensure-Venv -repoRoot $repoRoot
Ensure-Dependencies -venvPy $venvPy -repoRoot $repoRoot
Ensure-Workspace -repoRoot $repoRoot
Ensure-GeminiKey -repoRoot $repoRoot

& $venvPy -m src.main --agent
