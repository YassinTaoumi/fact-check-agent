@echo off
echo ========================================
echo  Stopping WhatsApp Fact-Checking Services
echo ========================================
echo.

echo Stopping Python processes related to fact-checking...

REM Kill specific Python processes by searching for script names
echo Stopping processing_storage_api.py...
taskkill /F /FI "WINDOWTITLE eq Processing Storage API*" >nul 2>&1

echo Stopping main_combined.py...
taskkill /F /FI "WINDOWTITLE eq Main Combined API*" >nul 2>&1

echo Stopping RabbitMQ workers...
taskkill /F /FI "WINDOWTITLE eq RabbitMQ Text Workers*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq RabbitMQ Fact-Check Workers*" >nul 2>&1

echo Stopping results consumer...
taskkill /F /FI "WINDOWTITLE eq Results Consumer*" >nul 2>&1

echo.
echo Stopping RabbitMQ Docker container (optional)...
docker stop rabbitmq_server >nul 2>&1
docker rm rabbitmq_server >nul 2>&1

echo.
echo ========================================
echo  All services stopped!
echo ========================================
echo.
pause
