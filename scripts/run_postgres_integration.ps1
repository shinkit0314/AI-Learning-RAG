param(
    [string]$DatabaseHost = "127.0.0.1",
    [int]$DatabasePort = 5432
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$migratorUser = "ai_learning_rag_migrator"
$projectDatabase = "ai_learning_rag"
$testDatabase = "ai_learning_rag_test"
$appUser = "ai_learning_rag_app"

function Invoke-CheckedCommand {
    param(
        [string]$Label,
        [string[]]$Arguments
    )

    Write-Host $Label
    & uv @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE."
    }
}

$secureMigratorPassword = Read-Host "Password for $migratorUser" -AsSecureString
$secureAppPassword = Read-Host "Password for $appUser" -AsSecureString
$migratorPasswordPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR(
    $secureMigratorPassword
)
$appPasswordPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureAppPassword)

try {
    $plainMigratorPassword = [Runtime.InteropServices.Marshal]::PtrToStringBSTR(
        $migratorPasswordPointer
    )
    $plainAppPassword = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($appPasswordPointer)
    $encodedMigratorPassword = [Uri]::EscapeDataString($plainMigratorPassword)
    $encodedAppPassword = [Uri]::EscapeDataString($plainAppPassword)
    $migratorPrefix = "postgresql+psycopg://${migratorUser}:${encodedMigratorPassword}@${DatabaseHost}:${DatabasePort}"
    $appPrefix = "postgresql+psycopg://${appUser}:${encodedAppPassword}@${DatabaseHost}:${DatabasePort}"
    $projectMigratorUrl = "${migratorPrefix}/${projectDatabase}"
    $testMigratorUrl = "${migratorPrefix}/${testDatabase}"
    $testAppUrl = "${appPrefix}/${testDatabase}"

    Push-Location $projectRoot
    try {
        $env:AI_LEARNING_RAG_DATABASE_URL = $projectMigratorUrl
        Invoke-CheckedCommand `
            -Label "[1/3] Migrating $projectDatabase..." `
            -Arguments @("run", "--no-sync", "alembic", "upgrade", "head")

        $env:AI_LEARNING_RAG_DATABASE_URL = $testMigratorUrl
        Invoke-CheckedCommand `
            -Label "[2/3] Migrating $testDatabase..." `
            -Arguments @("run", "--no-sync", "alembic", "upgrade", "head")

        $env:AI_LEARNING_RAG_TEST_DATABASE_URL = $testAppUrl
        $env:AI_LEARNING_RAG_RUN_POSTGRES_INTEGRATION = "1"
        Invoke-CheckedCommand `
            -Label "[3/3] Running explicit PostgreSQL integration tests..." `
            -Arguments @("run", "--no-sync", "pytest", "-m", "postgres_integration")
    }
    finally {
        Pop-Location
    }
}
finally {
    if ($migratorPasswordPointer -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($migratorPasswordPointer)
    }
    if ($appPasswordPointer -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($appPasswordPointer)
    }
    $plainMigratorPassword = $null
    $plainAppPassword = $null
    $encodedMigratorPassword = $null
    $encodedAppPassword = $null
    $projectMigratorUrl = $null
    $testMigratorUrl = $null
    $testAppUrl = $null
    Remove-Item -ErrorAction SilentlyContinue -LiteralPath Env:AI_LEARNING_RAG_DATABASE_URL
    Remove-Item -ErrorAction SilentlyContinue -LiteralPath Env:AI_LEARNING_RAG_TEST_DATABASE_URL
    Remove-Item -ErrorAction SilentlyContinue -LiteralPath Env:AI_LEARNING_RAG_RUN_POSTGRES_INTEGRATION
}
