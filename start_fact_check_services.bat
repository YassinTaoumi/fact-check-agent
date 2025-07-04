@echo off
echo ========================================
echo  WhatsApp Fact-Checking Pipeline Startup
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if RabbitMQ Docker container is running
echo Checking RabbitMQ status...
docker ps | findstr "rabbitmq" >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: RabbitMQ container not found running
    echo Starting RabbitMQ container...
    docker run -d --name rabbitmq_server -p 5672:5672 -p 15672:15672 rabbitmq:3-management
    echo Waiting 10 seconds for RabbitMQ to start...
    timeout /t 10 /nobreak >nul
)

echo.
echo Starting WhatsApp Fact-Checking Services...
echo.

REM Start processing_storage_api.py
echo [1/5] Starting Processing Storage API (port 8001)...
start "Processing Storage API" cmd /k "python processing_storage_api.py"
timeout /t 3 /nobreak >nul

REM Start main_combined.py
echo [2/5] Starting Main Combined API (port 8000)...
start "Main Combined API" cmd /k "python main_combined.py"
timeout /t 3 /nobreak >nul

REM Start rabbitmq_workers for text processing
echo [3/5] Starting RabbitMQ Text Workers...
start "RabbitMQ Text Workers" cmd /k "python rabbitmq_workers.py text"
timeout /t 2 /nobreak >nul

REM Start rabbitmq_workers for fact checking
echo [4/5] Starting RabbitMQ Fact-Check Workers...
start "RabbitMQ Fact-Check Workers" cmd /k "python rabbitmq_workers.py fact_check"
timeout /t 2 /nobreak >nul

REM Start results_consumer
echo [5/5] Starting Results Consumer...
start "Results Consumer" cmd /k "python results_consumer.py"
timeout /t 2 /nobreak >nul

echo.
echo ========================================
echo  All services started successfully!
echo ========================================
echo.
echo Services running:
echo   - Processing Storage API: http://localhost:8001
echo   - Main Combined API: http://localhost:8000
echo   - RabbitMQ Text Workers: Background
echo   - RabbitMQ Fact-Check Workers: Background
echo   - Results Consumer: Background
echo   - RabbitMQ Management: http://localhost:15672
echo.
echo To stop all services, close all command windows or press Ctrl+C in each.
echo.

REM Wait for user input before closing
echo Press any key to check service status...
pause >nul

REM Check if services are responding
echo.
echo Checking service health...
echo.

REM Test Processing Storage API
echo Testing Processing Storage API...
curl -s http://localhost:8001/health >nul 2>&1
if %errorlevel% equ 0 (
    echo   ✓ Processing Storage API - HEALTHY
) else (
    echo   ✗ Processing Storage API - NOT RESPONDING
)

REM Test Main Combined API
echo Testing Main Combined API...
curl -s http://localhost:8000/api/health >nul 2>&1
if %errorlevel% equ 0 (
    echo   ✓ Main Combined API - HEALTHY
) else (
    echo   ✗ Main Combined API - NOT RESPONDING
)

REM Test RabbitMQ
echo Testing RabbitMQ...
curl -s http://localhost:15672 >nul 2>&1
if %errorlevel% equ 0 (
    echo   ✓ RabbitMQ Management - HEALTHY
) else (
    echo   ✗ RabbitMQ Management - NOT RESPONDING
)

echo.
echo Service startup complete!
echo Check the individual command windows for detailed logs.
pause
