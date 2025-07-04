# WhatsApp Fact-Checking Pipeline Startup Script
# PowerShell version with better error handling and logging

param(
    [switch]$NoRabbitMQ,  # Skip RabbitMQ startup check
    [switch]$Verbose      # Enable verbose output
)

# Set up error handling
$ErrorActionPreference = "Continue"

# Function to write colored output
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

# Function to check if a port is listening
function Test-Port {
    param(
        [string]$ComputerName = "localhost",
        [int]$Port,
        [int]$Timeout = 3000
    )
    
    try {
        $tcpClient = New-Object System.Net.Sockets.TcpClient
        $asyncResult = $tcpClient.BeginConnect($ComputerName, $Port, $null, $null)
        $wait = $asyncResult.AsyncWaitHandle.WaitOne($Timeout, $false)
        
        if ($wait) {
            $tcpClient.EndConnect($asyncResult)
            $tcpClient.Close()
            return $true
        } else {
            $tcpClient.Close()
            return $false
        }
    } catch {
        return $false
    }
}

# Function to start a service in a new window
function Start-ServiceWindow {
    param(
        [string]$Title,
        [string]$Command,
        [string]$WorkingDirectory = (Get-Location).Path
    )
    
    $processArgs = @{
        FilePath = "powershell.exe"
        ArgumentList = "-NoExit", "-Command", "Set-Location '$WorkingDirectory'; $Command"
        WindowStyle = "Normal"
        PassThru = $true
    }
    
    try {
        $process = Start-Process @processArgs
        Write-ColorOutput "✓ Started: $Title (PID: $($process.Id))" "Green"
        return $process
    } catch {
        Write-ColorOutput "✗ Failed to start: $Title - $($_.Exception.Message)" "Red"
        return $null
    }
}

# Main script
Write-ColorOutput "========================================" "Cyan"
Write-ColorOutput " WhatsApp Fact-Checking Pipeline Startup" "Cyan"
Write-ColorOutput "========================================" "Cyan"
Write-Host ""

# Check Python
Write-ColorOutput "Checking Python installation..." "Yellow"
try {
    $pythonVersion = python --version 2>&1
    Write-ColorOutput "✓ Python found: $pythonVersion" "Green"
} catch {
    Write-ColorOutput "✗ ERROR: Python is not installed or not in PATH" "Red"
    Read-Host "Press Enter to exit"
    exit 1
}

# Check/Start RabbitMQ if not skipped
if (-not $NoRabbitMQ) {
    Write-ColorOutput "Checking RabbitMQ Docker container..." "Yellow"
    
    $rabbitMQRunning = docker ps --filter "name=rabbitmq" --format "table {{.Names}}" 2>$null | Select-String "rabbitmq"
    
    if (-not $rabbitMQRunning) {
        Write-ColorOutput "RabbitMQ container not found. Starting..." "Yellow"
        try {
            docker run -d --name rabbitmq_server -p 5672:5672 -p 15672:15672 rabbitmq:3-management 2>$null
            Write-ColorOutput "✓ RabbitMQ container started" "Green"
            Write-ColorOutput "Waiting 15 seconds for RabbitMQ to initialize..." "Yellow"
            Start-Sleep -Seconds 15
        } catch {
            Write-ColorOutput "⚠ Warning: Could not start RabbitMQ container - $($_.Exception.Message)" "Yellow"
        }
    } else {
        Write-ColorOutput "✓ RabbitMQ container is already running" "Green"
    }
}

Write-Host ""
Write-ColorOutput "Starting WhatsApp Fact-Checking Services..." "Cyan"
Write-Host ""

# Array to track started processes
$startedProcesses = @()

# Start services with delays
$services = @(
    @{ Name = "Processing Storage API"; Command = "python processing_storage_api.py"; Delay = 3 },
    @{ Name = "Main Combined API"; Command = "python main_combined.py"; Delay = 3 },
    @{ Name = "RabbitMQ Text Workers"; Command = "python rabbitmq_workers.py text"; Delay = 2 },
    @{ Name = "RabbitMQ Fact-Check Workers"; Command = "python rabbitmq_workers.py fact_check"; Delay = 2 },
    @{ Name = "Results Consumer"; Command = "python results_consumer.py"; Delay = 2 }
)

for ($i = 0; $i -lt $services.Count; $i++) {
    $service = $services[$i]
    Write-ColorOutput "[$($i+1)/$($services.Count)] Starting $($service.Name)..." "Yellow"
    
    $process = Start-ServiceWindow -Title $service.Name -Command $service.Command
    if ($process) {
        $startedProcesses += @{ Name = $service.Name; Process = $process }
    }
    
    Start-Sleep -Seconds $service.Delay
}

Write-Host ""
Write-ColorOutput "========================================" "Green"
Write-ColorOutput " All services startup initiated!" "Green"
Write-ColorOutput "========================================" "Green"
Write-Host ""

Write-ColorOutput "Services:" "Cyan"
Write-ColorOutput "  - Processing Storage API: http://localhost:8001" "White"
Write-ColorOutput "  - Main Combined API: http://localhost:8000" "White"
Write-ColorOutput "  - RabbitMQ Text Workers: Background Process" "White"
Write-ColorOutput "  - RabbitMQ Fact-Check Workers: Background Process" "White"
Write-ColorOutput "  - Results Consumer: Background Process" "White"
Write-ColorOutput "  - RabbitMQ Management: http://localhost:15672" "White"
Write-Host ""

# Wait a moment for services to start
Write-ColorOutput "Waiting 10 seconds for services to initialize..." "Yellow"
Start-Sleep -Seconds 10

# Health checks
Write-Host ""
Write-ColorOutput "Performing health checks..." "Cyan"
Write-Host ""

$healthChecks = @(
    @{ Name = "Processing Storage API"; Url = "http://localhost:8001/health"; Port = 8001 },
    @{ Name = "Main Combined API"; Url = "http://localhost:8000/api/health"; Port = 8000 },
    @{ Name = "RabbitMQ Management"; Url = "http://localhost:15672"; Port = 15672 },
    @{ Name = "RabbitMQ AMQP"; Url = ""; Port = 5672 }
)

foreach ($check in $healthChecks) {
    Write-Host "Testing $($check.Name)..." -NoNewline
    
    if ($check.Url) {
        try {
            $response = Invoke-WebRequest -Uri $check.Url -TimeoutSec 5 -UseBasicParsing -ErrorAction SilentlyContinue
            if ($response.StatusCode -eq 200) {
                Write-ColorOutput " ✓ HEALTHY" "Green"
            } else {
                Write-ColorOutput " ✗ HTTP $($response.StatusCode)" "Red"
            }
        } catch {
            Write-ColorOutput " ✗ NOT RESPONDING" "Red"
        }
    } else {
        # Port check for RabbitMQ AMQP
        if (Test-Port -Port $check.Port) {
            Write-ColorOutput " ✓ LISTENING" "Green"
        } else {
            Write-ColorOutput " ✗ NOT LISTENING" "Red"
        }
    }
}

Write-Host ""
Write-ColorOutput "========================================" "Green"
Write-ColorOutput " Service Health Check Complete" "Green"
Write-ColorOutput "========================================" "Green"
Write-Host ""

Write-ColorOutput "Started Processes:" "Cyan"
foreach ($proc in $startedProcesses) {
    $status = if ($proc.Process.HasExited) { "EXITED" } else { "RUNNING" }
    $color = if ($proc.Process.HasExited) { "Red" } else { "Green" }
    Write-ColorOutput "  - $($proc.Name): $status (PID: $($proc.Process.Id))" $color
}

Write-Host ""
Write-ColorOutput "Pipeline startup complete!" "Green"
Write-ColorOutput "Check individual PowerShell windows for detailed service logs." "Yellow"
Write-ColorOutput "To stop all services, close the individual PowerShell windows or use Task Manager." "Yellow"

Write-Host ""
Read-Host "Press Enter to exit this script (services will continue running)"
