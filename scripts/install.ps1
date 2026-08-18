[CmdletBinding()]
param(
    [string]$InstallRoot = (Join-Path $env:LOCALAPPDATA "PLPredict"),
    [string]$RepositoryUrl = "https://github.com/kianjindal2010/pl-predict.git"
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$Message) {
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Ensure-Command([string]$Command, [string]$WingetId) {
    if (Get-Command $Command -ErrorAction SilentlyContinue) { return }
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "Windows Package Manager (winget) is required to install $Command. Install App Installer from Microsoft Store and run this again."
    }
    Write-Step "Installing $Command with winget"
    winget install --id $WingetId --exact --source winget --accept-package-agreements --accept-source-agreements
    if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
        throw "$Command was installed but is not available in this terminal. Close PowerShell, reopen it, and run the installer again."
    }
}

Write-Host "PL Predict installer" -ForegroundColor Magenta
Ensure-Command git "Git.Git"
Ensure-Command python "Python.Python.3.12"

Write-Step "Getting PL Predict"
if (Test-Path (Join-Path $InstallRoot ".git")) {
    git -C $InstallRoot pull --ff-only
} elseif (Test-Path $InstallRoot) {
    throw "$InstallRoot already exists but is not a PL Predict Git checkout. Rename it first; the installer will not delete local files."
} else {
    $parent = Split-Path -Parent $InstallRoot
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    git clone $RepositoryUrl $InstallRoot
}

Write-Step "Creating the local Python environment"
$python = Get-Command python -ErrorAction Stop
& $python.Source -m venv (Join-Path $InstallRoot ".venv")
$venvPython = Join-Path $InstallRoot ".venv\Scripts\python.exe"
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install --editable $InstallRoot

Write-Step "Creating the pl-predict command"
$launcherDir = Join-Path $InstallRoot "bin"
New-Item -ItemType Directory -Path $launcherDir -Force | Out-Null
$launcher = Join-Path $launcherDir "pl-predict.cmd"
@"
@echo off
"$InstallRoot\.venv\Scripts\pl-predict.exe" %*
"@ | Set-Content -Path $launcher -Encoding ascii

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$launcherDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$launcherDir;$userPath", "User")
}

Write-Step "Checking the installation"
& $venvPython -c "import pl_predict; print('PL Predict is ready')"
Write-Host "`nInstalled in $InstallRoot" -ForegroundColor Green
Write-Host "Open a new PowerShell or Command Prompt window, then run: pl-predict" -ForegroundColor Yellow
Write-Host "Each launch refreshes official FPL data, recalculates Hybrid xP, and opens the local dashboard." -ForegroundColor DarkGray
