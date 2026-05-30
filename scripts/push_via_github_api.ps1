param(
    [string]$RepoRoot = (Split-Path $PSScriptRoot -Parent),
    [string]$Owner = "kikuwa",
    [string]$Repo = "finRiskTool",
    [string]$Branch = "main",
    [string]$CommitMessage = "Refactor data_core modules and add CLI pipeline support",
    [int]$DelayMs = 500,
    [int]$MaxRetries = 5
)

$ErrorActionPreference = "Stop"
Set-Location $RepoRoot

$credLine = (Get-Content "$env:USERPROFILE\.git-credentials" | Select-String "github.com").ToString()
$token = $credLine -replace 'https://[^:]+:([^@]+)@github.com', '$1'
$proxy = "http://proxyhk.huawei.com:8080"

$headers = @{
    Authorization = "Bearer $token"
    Accept = "application/vnd.github+json"
    "X-GitHub-Api-Version" = "2022-11-28"
}

function Invoke-GitHubApi {
    param(
        [string]$Method = "Get",
        [string]$Uri,
        [object]$Body = $null,
        [int]$Retries = $MaxRetries
    )
    for ($attempt = 1; $attempt -le $Retries; $attempt++) {
        try {
            $params = @{
                Method = $Method
                Uri = $Uri
                Headers = $headers
                Proxy = $proxy
                ProxyUseDefaultCredentials = $true
            }
            if ($null -ne $Body) {
                $params.Body = ($Body | ConvertTo-Json -Depth 20 -Compress)
                $params.ContentType = "application/json"
            }
            return Invoke-RestMethod @params
        } catch {
            if ($attempt -eq $Retries) { throw }
            Start-Sleep -Milliseconds ($DelayMs * $attempt * 3)
        }
    }
}

function Get-LocalBlobSha {
    param([string]$FilePath)
    return (git hash-object $FilePath).Trim()
}

Write-Host "Loading remote file list..."
$ref = Invoke-GitHubApi -Uri "https://api.github.com/repos/$Owner/$Repo/git/ref/heads/$Branch"
$remoteCommit = Invoke-GitHubApi -Uri "https://api.github.com/repos/$Owner/$Repo/commits/$($ref.object.sha)"
$treeSha = $remoteCommit.commit.tree.sha
$remoteTree = Invoke-GitHubApi -Uri "https://api.github.com/repos/$Owner/$Repo/git/trees/${treeSha}?recursive=1"

$remoteBlobShas = @{}
$remotePaths = @{}
foreach ($item in $remoteTree.tree) {
    if ($item.type -eq "blob") {
        $remoteBlobShas[$item.path] = $item.sha
        $remotePaths[$item.path] = $true
    }
}

$localFiles = @(git ls-files | ForEach-Object { $_ -replace '\\', '/' })
$localSet = [System.Collections.Generic.HashSet[string]]::new([string[]]$localFiles)

$pending = @()
foreach ($apiPath in $localFiles) {
    $fullPath = Join-Path $RepoRoot ($apiPath -replace '/', '\')
    if (-not (Test-Path -LiteralPath $fullPath)) { continue }
    $localSha = Get-LocalBlobSha -FilePath $fullPath
    if (-not $remoteBlobShas.ContainsKey($apiPath) -or $remoteBlobShas[$apiPath] -ne $localSha) {
        $pending += $apiPath
    }
}

Write-Host "Pending uploads: $($pending.Count) / $($localFiles.Count)"
$i = 0
foreach ($apiPath in $pending) {
    $i++
    $fullPath = Join-Path $RepoRoot ($apiPath -replace '/', '\')
    $bytes = [System.IO.File]::ReadAllBytes($fullPath)
    $encoded = [Convert]::ToBase64String($bytes)
    $body = @{
        message = if ($i -eq $pending.Count) { $CommitMessage } else { "sync: $apiPath" }
        content = $encoded
        branch = $Branch
    }
    if ($remoteBlobShas.ContainsKey($apiPath)) {
        $body.sha = $remoteBlobShas[$apiPath]
    }

    $encodedPath = [Uri]::EscapeDataString($apiPath).Replace('%2F', '/')
    Invoke-GitHubApi -Method Put -Uri "https://api.github.com/repos/$Owner/$Repo/contents/$encodedPath" -Body $body | Out-Null
    Write-Host "  [$i/$($pending.Count)] $apiPath ($([math]::Round($bytes.Length/1KB,1)) KB)"
    $remoteBlobShas[$apiPath] = Get-LocalBlobSha -FilePath $fullPath
    Start-Sleep -Milliseconds $DelayMs
}

$toDelete = $remotePaths.Keys | Where-Object { -not $localSet.Contains($_) }
if ($toDelete.Count -gt 0) {
    Write-Host "Deleting $($toDelete.Count) removed remote files..."
    foreach ($apiPath in $toDelete) {
        if (-not $remoteBlobShas.ContainsKey($apiPath)) { continue }
        $encodedPath = [Uri]::EscapeDataString($apiPath).Replace('%2F', '/')
        $body = @{
            message = "remove: $apiPath"
            sha = $remoteBlobShas[$apiPath]
            branch = $Branch
        }
        Invoke-GitHubApi -Method Delete -Uri "https://api.github.com/repos/$Owner/$Repo/contents/$encodedPath" -Body $body | Out-Null
        Write-Host "  deleted $apiPath"
        Start-Sleep -Milliseconds $DelayMs
    }
}

Write-Host "Done. https://github.com/$Owner/$Repo/tree/$Branch"
