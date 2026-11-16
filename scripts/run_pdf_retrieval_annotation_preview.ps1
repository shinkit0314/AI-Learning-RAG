param(
    [string]$DatabaseHost = "127.0.0.1",
    [int]$DatabasePort = 5432
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$appUser = "ai_learning_rag_app"
$database = "ai_learning_rag"
$outputDir = "output/pdf-retrieval-annotation-preview"
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
        & uv run --no-sync python scripts/pdf_indexing_preview.py `
            --manifest configs/pdf-retrieval-annotation-preview-v1.json `
            --output-dir $outputDir
        if ($LASTEXITCODE -ne 0) {
            throw "PDF retrieval annotation preview failed with exit code $LASTEXITCODE."
        }

        & uv run --no-sync python scripts/build_retrieval_annotation_review.py `
            --preview "$outputDir/retrieval-preview.json" `
            --output "$outputDir/annotation-review.md"
        if ($LASTEXITCODE -ne 0) {
            throw "Annotation review rendering failed with exit code $LASTEXITCODE."
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
