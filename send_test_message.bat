@echo off
REM Quick WhatsApp Test Message Sender
REM Wrapper script for send_test_message.py

echo ========================================
echo  WhatsApp Test Message Sender
echo ========================================
echo.

REM Check if Python script exists
if not exist "send_test_message.py" (
    echo ERROR: send_test_message.py not found
    pause
    exit /b 1
)

REM Show available options
echo Available message types:
echo   1. simple - Basic test message
echo   2. fact_check - Anti-aging misinformation
echo   3. misinformation - Vaccine conspiracy
echo   4. covid - COVID vaccine misinformation
echo   5. custom - Enter your own message
echo   6. check - Only check recent messages
echo.

REM Get user choice
set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" (
    python send_test_message.py --type simple
) else if "%choice%"=="2" (
    python send_test_message.py --type fact_check
) else if "%choice%"=="3" (
    python send_test_message.py --type misinformation
) else if "%choice%"=="4" (
    python send_test_message.py --type covid
) else if "%choice%"=="5" (
    set /p custom_msg="Enter your custom message: "
    python send_test_message.py --message "!custom_msg!"
) else if "%choice%"=="6" (
    python send_test_message.py --check-only
) else (
    echo Invalid choice. Please run the script again.
    pause
    exit /b 1
)

echo.
pause
