param(
    [Parameter(Mandatory = $true)]
    [string]$VGMPlayExe,

    [Parameter(Mandatory = $true)]
    [string]$InputDir,

    [Parameter(Mandatory = $true)]
    [string]$OutputDir,

    [int]$TimeoutSeconds = 600,

    [int]$PostWaitSeconds = 10,

    [switch]$Overwrite,

    [switch]$PreserveSubdirs,

    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Cyan
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Fail {
    param([string]$Message)
    Write-Host "[FAIL] $Message" -ForegroundColor Red
}

function Get-RelativePath {
    param(
        [string]$BasePath,
        [string]$FullPath
    )

    $base = [System.IO.Path]::GetFullPath($BasePath).TrimEnd('\') + '\'
    $full = [System.IO.Path]::GetFullPath($FullPath)

    $baseUri = New-Object System.Uri($base)
    $fullUri = New-Object System.Uri($full)

    return [System.Uri]::UnescapeDataString(
        $baseUri.MakeRelativeUri($fullUri).ToString()
    ).Replace('/', '\')
}

function Wait-ForStableFile {
    param(
        [string]$Path,
        [int]$MaxWaitSeconds = 30,
        [int]$StableChecks = 3
    )

    $lastSize = -1
    $stableCount = 0
    $start = Get-Date

    while (((Get-Date) - $start).TotalSeconds -lt $MaxWaitSeconds) {
        if (Test-Path $Path) {
            $item = Get-Item $Path
            $size = $item.Length

            if ($size -gt 0 -and $size -eq $lastSize) {
                $stableCount++
            }
            else {
                $stableCount = 0
            }

            if ($stableCount -ge $StableChecks) {
                return $true
            }

            $lastSize = $size
        }

        Start-Sleep -Seconds 1
    }

    return $false
}

function Invoke-VGMPlay {
    param(
        [string]$ExePath,
        [string]$InputFile,
        [string]$WorkingDirectory,
        [int]$TimeoutSeconds
    )

    # IMPORTANT:
    # No stdout/stderr redirection here.
    # Redirected output can deadlock longer conversions in Windows PowerShell.
    $process = Start-Process `
        -FilePath $ExePath `
        -ArgumentList "`"$InputFile`"" `
        -WorkingDirectory $WorkingDirectory `
        -PassThru `
        -WindowStyle Hidden

    $finished = $process.WaitForExit($TimeoutSeconds * 1000)

    if (-not $finished) {
        try {
            $process.Kill()
        }
        catch {
            Write-Warn "Could not kill VGMPlay process cleanly."
        }

        return @{
            TimedOut = $true
            ExitCode = -999
        }
    }

    return @{
        TimedOut = $false
        ExitCode = $process.ExitCode
    }
}

if (!(Test-Path $VGMPlayExe)) {
    throw "VGMPlay executable not found: $VGMPlayExe"
}

if (!(Test-Path $InputDir)) {
    throw "Input directory not found: $InputDir"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$vgmPlayDir = Split-Path $VGMPlayExe -Parent
$logPath = Join-Path $OutputDir "vgm_render_log.csv"

if (!(Test-Path $logPath)) {
    "status,input,output,message" | Out-File -FilePath $logPath -Encoding utf8
}

$files = Get-ChildItem -Path $InputDir -Recurse -File |
    Where-Object {
        $_.Extension.ToLowerInvariant() -eq ".vgm" -or
        $_.Extension.ToLowerInvariant() -eq ".vgz"
    } |
    Sort-Object FullName

Write-Info "VGMPlay: $VGMPlayExe"
Write-Info "VGMPlay working dir: $vgmPlayDir"
Write-Info "InputDir: $InputDir"
Write-Info "OutputDir: $OutputDir"
Write-Info "Files found: $($files.Count)"
Write-Info "Timeout per file: $TimeoutSeconds seconds"

$success = 0
$skipped = 0
$failed = 0

foreach ($file in $files) {
    $relative = Get-RelativePath -BasePath $InputDir -FullPath $file.FullName
    $relativeDir = [System.IO.Path]::GetDirectoryName($relative)
    $relativeBase = [System.IO.Path]::GetFileNameWithoutExtension($relative)

    if ($PreserveSubdirs -and -not [string]::IsNullOrWhiteSpace($relativeDir)) {
        $outPath = Join-Path (Join-Path $OutputDir $relativeDir) ($relativeBase + ".wav")
    }
    else {
        $safeName = $relative.Replace('\', '__').Replace('/', '__')
        $safeBase = [System.IO.Path]::GetFileNameWithoutExtension($safeName)
        $outPath = Join-Path $OutputDir ($safeBase + ".wav")
    }

    $outDir = Split-Path $outPath -Parent
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null

    if ((Test-Path $outPath) -and (-not $Overwrite)) {
        Write-Info "Skipping existing: $outPath"
        "skipped,""$($file.FullName)"",""$outPath"",""Already exists""" | Add-Content -Path $logPath -Encoding utf8
        $skipped++
        continue
    }

    $generatedWav = Join-Path $file.DirectoryName ([System.IO.Path]::GetFileNameWithoutExtension($file.Name) + ".wav")

    if ((Test-Path $generatedWav) -and $Overwrite) {
        Remove-Item $generatedWav -Force
    }

    if ((Test-Path $outPath) -and $Overwrite) {
        Remove-Item $outPath -Force
    }

    Write-Info "Rendering: $($file.FullName)"
    Write-Info "Generated source WAV expected at: $generatedWav"
    Write-Info "Final output: $outPath"

    if ($DryRun) {
        "dryrun,""$($file.FullName)"",""$outPath"",""Dry run""" | Add-Content -Path $logPath -Encoding utf8
        continue
    }

    $result = Invoke-VGMPlay `
        -ExePath $VGMPlayExe `
        -InputFile $file.FullName `
        -WorkingDirectory $vgmPlayDir `
        -TimeoutSeconds $TimeoutSeconds

    if ($result.TimedOut) {
        Write-Warn "Timed out: $($file.FullName)"
    }

    # Some VGMPlay writes finalize shortly after the process exits or is killed.
    $stable = Wait-ForStableFile `
        -Path $generatedWav `
        -MaxWaitSeconds $PostWaitSeconds `
        -StableChecks 3

    if (-not $stable) {
        Write-Fail "No stable WAV produced: $generatedWav"
        "failed,""$($file.FullName)"",""$outPath"",""No stable WAV produced. TimedOut=$($result.TimedOut), ExitCode=$($result.ExitCode)""" | Add-Content -Path $logPath -Encoding utf8
        $failed++
        continue
    }

    try {
        Move-Item -Path $generatedWav -Destination $outPath -Force

        if (Test-Path $outPath) {
            $outItem = Get-Item $outPath

            if ($outItem.Length -le 44) {
                Write-Fail "Output WAV is empty or header-only: $outPath"
                "failed,""$($file.FullName)"",""$outPath"",""Output WAV too small: $($outItem.Length) bytes""" | Add-Content -Path $logPath -Encoding utf8
                $failed++
                continue
            }

            Write-Info "Saved: $outPath"
            "success,""$($file.FullName)"",""$outPath"",""OK""" | Add-Content -Path $logPath -Encoding utf8
            $success++
        }
        else {
            Write-Fail "Move appeared to fail: $outPath"
            "failed,""$($file.FullName)"",""$outPath"",""Move failed""" | Add-Content -Path $logPath -Encoding utf8
            $failed++
        }
    }
    catch {
        Write-Fail "Failed moving WAV: $($_.Exception.Message)"
        "failed,""$($file.FullName)"",""$outPath"",""$($_.Exception.Message)""" | Add-Content -Path $logPath -Encoding utf8
        $failed++
    }
}

Write-Host ""
Write-Info "Done."
Write-Info "Success: $success"
Write-Info "Skipped: $skipped"
Write-Info "Failed:  $failed"
Write-Info "Log:     $logPath"
