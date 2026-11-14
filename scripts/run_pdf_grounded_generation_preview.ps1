param(
    [string]$DatabaseHost = "127.0.0.1",
    [int]$DatabasePort = 5432
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$appUser = "ai_learning_rag_app"
$database = "ai_learning_rag"

Write-Host "This explicit preview sends selected PDF evidence to DeepSeek."
Write-Host "DEEPSEEK_API_KEY is read from the current session or the project-root .env file."
Write-Host "It writes local JSON and Markdown review artifacts and may incur API charges."
$securePassword = Read-Host "Password for $appUser" -AsSecureString
$passwordPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)

try {
    $plainPassword = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPointer)
    $encodedPassword = [Uri]::EscapeDataString($plainPassword)
    $env:AI_LEARNING_RAG_DATABASE_URL = (
        "postgresql+psycopg://${appUser}:${encodedPassword}@${DatabaseHost}:${DatabasePort}/${database}"
    )

    Push-Location $projectRoot
    try {
        & uv run --no-sync python scripts/pdf_grounded_generation_preview.py
        if ($LASTEXITCODE -ne 0) {
            throw "PDF grounded-generation preview failed with exit code $LASTEXITCODE."
        }
    }
    finally {
        Pop-Location
    }
}
finally {
    if ($passwordPointer -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($passwordPointer)
    }
    $plainPassword = $null
    $encodedPassword = $null
    Remove-Item -ErrorAction SilentlyContinue -LiteralPath Env:AI_LEARNING_RAG_DATABASE_URL
}
