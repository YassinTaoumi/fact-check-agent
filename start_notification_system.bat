@echo off
echo 🚀 Starting WhatsApp Fact-Check Notification System
echo ====================================================

REM Check if Go is installed
go version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Go is not installed or not in PATH
    echo Please install Go from https://golang.org/dl/
    pause
    exit /b 1
)

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    pause
    exit /b 1
)

echo ✅ Prerequisites check passed

REM Create logs directory
if not exist "logs" mkdir logs

echo.
echo 📱 Starting WhatsApp API Server...
echo    This will handle sending notification messages

REM Start WhatsApp API server in background
start "WhatsApp API Server" /MIN cmd /c "go run whatsapp_api_server.go > logs\whatsapp_api.log 2>&1"

REM Wait a moment for the server to start
timeout /t 3 /nobreak > nul

echo.
echo 🔍 Starting Fact-Checking Worker...
echo    This will process completed messages and send notifications

REM Start fact-checking worker in background  
start "Fact-Check Worker" /MIN cmd /c "python rabbitmq_workers.py fact_check > logs\fact_check_worker.log 2>&1"

echo.
echo 📊 Starting Results Consumer...
echo    This monitors processing completion and triggers fact-checking

REM Start results consumer in background
start "Results Consumer" /MIN cmd /c "python results_consumer.py > logs\results_consumer.log 2>&1"

echo.
echo ✅ All services started!
echo.
echo 📋 Service Status:
echo    - WhatsApp API Server: http://localhost:9090/health
echo    - Fact-Check Worker: Running in background
echo    - Results Consumer: Running in background
echo.
echo 📁 Logs Location:
echo    - WhatsApp API: logs\whatsapp_api.log
echo    - Fact-Check Worker: logs\fact_check_worker.log  
echo    - Results Consumer: logs\results_consumer.log
echo.
echo 🔍 Testing the notification system...

REM Test the notification system
python test_notification_system.py

echo.
echo 📱 To manually test WhatsApp sending:
echo    curl -X POST http://localhost:9090/send-message ^
echo         -H "Content-Type: application/json" ^
echo         -d "{\"recipient\":\"1234567890\",\"message\":\"Test message\"}"
echo.
echo ⚠️  Remember:
echo    1. Make sure WhatsApp is connected (scan QR code if needed)
echo    2. Replace phone numbers in tests with real numbers
echo    3. Check logs if notifications aren't working
echo.
echo Press any key to view live logs or Ctrl+C to exit...
pause > nul

REM Show live logs
echo.
echo 📊 Live Logs (Press Ctrl+C to exit):
powershell -Command "Get-Content logs\fact_check_worker.log -Wait"
