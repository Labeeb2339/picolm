[CmdletBinding()]
param(
    [switch]$SkipHellaSwag
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo ".venv\Scripts\python.exe"
$userProfile = [Environment]::GetFolderPath("UserProfile")
$repoPattern = [regex]::Escape($repo)
$profilePattern = [regex]::Escape($userProfile)
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Missing PicoLM virtual environment: $python"
}

Set-Location -LiteralPath $repo
$gitStatus = & git status --porcelain=v1 --untracked-files=all
if ($LASTEXITCODE -ne 0) {
    throw "Unable to inspect the Git worktree"
}
if ($gitStatus) {
    throw "GPU evidence requires a committed, clean worktree"
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$evidenceDir = Join-Path $repo "out\evidence\$stamp"
New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null

$sourceLog = Join-Path $evidenceDir "00-source.txt"
@(
    "commit=$((& git rev-parse HEAD).Trim())"
    "tree=$((& git rev-parse 'HEAD^{tree}').Trim())"
) | Set-Content -LiteralPath $sourceLog -Encoding utf8

function Invoke-Evidence {
    param(
        [Parameter(Mandatory)] [string]$Name,
        [Parameter(Mandatory)] [string[]]$CommandArgs
    )

    $log = Join-Path $evidenceDir "$Name.txt"
    "# .\.venv\Scripts\python.exe $($CommandArgs -join ' ')" |
        Tee-Object -FilePath $log
    & $python @CommandArgs 2>&1 |
        ForEach-Object {
            $line = "$_"
            $line = $line -replace $repoPattern, "<repo>"
            $line = $line -replace $profilePattern, "<home>"
            $line
        } |
        Tee-Object -FilePath $log -Append
    $pythonExitCode = $LASTEXITCODE
    if ($pythonExitCode -ne 0) {
        throw "Evidence step failed ($pythonExitCode): $Name"
    }
}

$gpuLog = Join-Path $evidenceDir "01-gpu.txt"
nvidia-smi `
    --query-gpu=name,driver_version,memory.total,compute_cap `
    --format=csv,noheader |
    Tee-Object -FilePath $gpuLog
if ($LASTEXITCODE -ne 0) {
    throw "nvidia-smi failed"
}

Invoke-Evidence "01-artifacts" @(
    "scripts\verify_artifacts.py",
    "--require-all"
)
Invoke-Evidence "02-flash-correctness" @(
    "-m", "pytest", "-q", "tests\test_flash_attn.py",
    "--basetemp", ".pytest-tmp\gpu-flash"
)
Invoke-Evidence "03-perplexity" @(
    "-m", "picolm", "eval",
    "--ckpt", "out\ckpt.pt",
    "--text", "data\input.txt",
    "--num-batches", "100"
)
Invoke-Evidence "04-kv-and-int8" @(
    "-m", "picolm", "benchmark",
    "--ckpt", "out\ckpt.pt",
    "--text", "data\input.txt",
    "--max-tokens", "200"
)
Invoke-Evidence "05-attention-benchmark" @(
    "scripts\bench_attention.py"
)

if (-not $SkipHellaSwag) {
    Invoke-Evidence "06-hellaswag" @(
        "scripts\run_hellaswag.py",
        "--ckpt", "out\ckpt.pt",
        "--data", "data\hellaswag_val.jsonl",
        "--limit", "2000"
    )
}

$receipt = "out\evidence\$stamp\receipt.json"
Invoke-Evidence "07-receipt" @(
    "scripts\repro_receipt.py",
    "--output", $receipt,
    "--require-artifacts"
)

$checksumPath = Join-Path $evidenceDir "SHA256SUMS.txt"
Get-ChildItem -LiteralPath $evidenceDir -File |
    Sort-Object Name |
    ForEach-Object {
        $hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLower()
        "$hash  $($_.Name)"
    } | Set-Content -LiteralPath $checksumPath -Encoding utf8

Write-Output "GPU evidence complete: $evidenceDir"
Write-Output "Source/environment receipt: $receipt"
