# WhatsApp Fact-Checking Pipeline Stop Script

param(
    [switch]$StopRabbitMQ  # Also stop RabbitMQ container
)

function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

Write-ColorOutput "========================================" "Red"
Write-ColorOutput " Stopping WhatsApp Fact-Checking Services" "Red"
Write-ColorOutput "========================================" "Red"
Write-Host ""

# Get all Python processes that might be our services
$pythonProcesses = Get-Process python -ErrorAction SilentlyContinue

if ($pythonProcesses) {
    Write-ColorOutput "Found $($pythonProcesses.Count) Python processes. Checking for fact-checking services..." "Yellow"
    
    $serviceProcesses = @()
    
    foreach ($proc in $pythonProcesses) {
        try {
            $commandLine = (Get-WmiObject Win32_Process -Filter "ProcessId = $($proc.Id)").CommandLine
            
            if ($commandLine -and (
                $commandLine -like "*processing_storage_api.py*" -or
                $commandLine -like "*main_combined.py*" -or
                $commandLine -like "*rabbitmq_workers.py*" -or
                $commandLine -like "*results_consumer.py*"
            )) {
                $serviceProcesses += $proc
                Write-ColorOutput "  Found service process: PID $($proc.Id) - $commandLine" "Yellow"
            }
        } catch {
            # Skip processes we can't access
        }
    }
    
    if ($serviceProcesses.Count -gt 0) {
        Write-ColorOutput "Stopping $($serviceProcesses.Count) fact-checking service processes..." "Red"
        
        foreach ($proc in $serviceProcesses) {
            try {
                Stop-Process -Id $proc.Id -Force
                Write-ColorOutput "  ✓ Stopped process PID $($proc.Id)" "Green"
            } catch {
                Write-ColorOutput "  ✗ Failed to stop process PID $($proc.Id)" "Red"
            }
        }
    } else {
        Write-ColorOutput "No fact-checking service processes found." "Green"
    }
} else {
    Write-ColorOutput "No Python processes found." "Green"
}

# Stop PowerShell windows with specific titles
Write-Host ""
Write-ColorOutput "Closing service windows..." "Yellow"

$windowTitles = @(
    "*Processing Storage API*",
    "*Main Combined API*", 
    "*RabbitMQ Text Workers*",
    "*RabbitMQ Fact-Check Workers*",
    "*Results Consumer*"
)

foreach ($title in $windowTitles) {
    $windows = Get-Process powershell -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowTitle -like $title }
    
    if ($windows) {
        foreach ($window in $windows) {
            try {
                Stop-Process -Id $window.Id -Force
                Write-ColorOutput "  ✓ Closed window: $($window.MainWindowTitle)" "Green"
            } catch {
                Write-ColorOutput "  ✗ Failed to close window: $($window.MainWindowTitle)" "Red"
            }
        }
    }
}

# Optionally stop RabbitMQ container
if ($StopRabbitMQ) {
    Write-Host ""
    Write-ColorOutput "Stopping RabbitMQ Docker container..." "Yellow"
    
    try {
        $result = docker stop rabbitmq_server 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "✓ RabbitMQ container stopped" "Green"
        } else {
            Write-ColorOutput "⚠ RabbitMQ container may not be running" "Yellow"
        }
        
        $result = docker rm rabbitmq_server 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "✓ RabbitMQ container removed" "Green"
        }
    } catch {
        Write-ColorOutput "✗ Error managing RabbitMQ container: $($_.Exception.Message)" "Red"
    }
}

Write-Host ""
Write-ColorOutput "========================================" "Green"
Write-ColorOutput " Service shutdown complete!" "Green"
Write-ColorOutput "========================================" "Green"
Write-Host ""

if (-not $StopRabbitMQ) {
    Write-ColorOutput "Note: RabbitMQ container is still running." "Yellow"
    Write-ColorOutput "Use '-StopRabbitMQ' parameter to also stop RabbitMQ." "Yellow"
}

Write-Host ""
Read-Host "Press Enter to exit"
