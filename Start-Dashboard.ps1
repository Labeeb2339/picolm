[CmdletBinding()]
param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 8501
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $repo '.venv\Scripts\python.exe'
$checkpoint = Join-Path $repo 'out\ckpt.pt'
$dashboard = Join-Path $repo 'src\picolm\demo.py'

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "PicoLM's local Python environment is missing: $python"
}
if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
    throw "The demo checkpoint is missing: $checkpoint"
}

Set-Location -LiteralPath $repo
& $python -m streamlit run $dashboard `
    --server.headless=false `
    --server.address=127.0.0.1 `
    "--server.port=$Port" `
    --browser.gatherUsageStats=false
exit $LASTEXITCODE
