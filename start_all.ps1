# StreamGuard - All-in-One Local MVP Startup Script
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "         StreamGuard Real-Time Platform Startup           " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Start Docker Services
Write-Host "`n[1/3] Starting Docker services (Kafka, Bronze, Silver, Gold)..." -ForegroundColor Yellow
docker compose up -d

# 2. Check Kafka Health
Write-Host "`n[2/3] Verifying Kafka KRaft broker..." -ForegroundColor Yellow
docker compose ps

# 3. Start Binance Producer in Background
Write-Host "`n[3/3] Launching Binance Trade Ingestion Producer..." -ForegroundColor Yellow
$producerScript = Join-Path $PSScriptRoot "ingestion\binance_producer.py"
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (Test-Path $venvPython) {
    Start-Process -FilePath $venvPython -ArgumentList $producerScript -WindowStyle Minimized
    Write-Host "Binance Producer launched in background with .venv." -ForegroundColor Green
} else {
    Start-Process -FilePath "python" -ArgumentList $producerScript -WindowStyle Minimized
    Write-Host "Binance Producer launched in background with system python." -ForegroundColor Green
}

# 4. Launch Plotly Dashboard Server
Write-Host "`nLaunching Plotly Dashboard on http://localhost:8050..." -ForegroundColor Cyan
$dashboardScript = Join-Path $PSScriptRoot "dashboard\app.py"
Start-Process -FilePath "python" -ArgumentList "$dashboardScript 8050" -WindowStyle Normal

Start-Sleep -Seconds 3
Start-Process "http://localhost:8050"

Write-Host "`nStreamGuard platform is active!" -ForegroundColor Green
Write-Host "Dashboard URL: http://localhost:8050" -ForegroundColor Green
Write-Host "HTML Report: file://$($PSScriptRoot -replace '\\','/')/dashboard/index.html" -ForegroundColor Green
