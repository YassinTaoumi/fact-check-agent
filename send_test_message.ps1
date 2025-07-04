# Send Test Message to WhatsApp Processing API
# PowerShell script to send test messages as if they came from 212613880290

param(
    [string]$MessageType = "simple",
    [string]$CustomMessage = "",
    [switch]$CheckOnly,
    [switch]$ListTypes,
    [switch]$Help
)

# Configuration
$PROCESSING_STORAGE_API = "http://localhost:8001"
$SENDER_PHONE = "212613880290"

# Test messages
$TEST_MESSAGES = @{
    "simple" = "Hello, this is a test message from 212613880290"
    "fact_check" = "BREAKING: Scientists at Harvard discovered that eating 5 bananas daily can completely reverse aging and add 50 years to your life. This simple fruit contains quantum proteins that repair DNA damage instantly!"
    "misinformation" = "URGENT: The government is putting mind control chips in vaccines. Share this message to warn everyone before it's too late!"
    "covid" = "COVID-19 vaccines contain 5G microchips that track your location and can control your thoughts remotely."
}

function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

function Show-Help {
    Write-ColorOutput "🚀 WhatsApp Test Message Sender" "Cyan"
    Write-ColorOutput "=" * 50 "Cyan"
    Write-Host ""
    Write-ColorOutput "USAGE:" "Yellow"
    Write-Host "  .\send_test_message.ps1 [-MessageType <type>] [-CustomMessage <text>] [-CheckOnly] [-ListTypes]"
    Write-Host ""
    Write-ColorOutput "PARAMETERS:" "Yellow"
    Write-Host "  -MessageType    : Type of predefined message (simple, fact_check, misinformation, covid)"
    Write-Host "  -CustomMessage  : Send a custom message instead of predefined one"
    Write-Host "  -CheckOnly      : Only check recent messages, don't send new one"
    Write-Host "  -ListTypes      : Show available message types"
    Write-Host "  -Help           : Show this help"
    Write-Host ""
    Write-ColorOutput "EXAMPLES:" "Yellow"
    Write-Host "  .\send_test_message.ps1"
    Write-Host "  .\send_test_message.ps1 -MessageType fact_check"
    Write-Host "  .\send_test_message.ps1 -CustomMessage `"My custom test message`""
    Write-Host "  .\send_test_message.ps1 -CheckOnly"
}

function Test-ApiHealth {
    Write-ColorOutput "🔍 Checking Processing Storage API health..." "Yellow"
    
    try {
        $response = Invoke-RestMethod -Uri "$PROCESSING_STORAGE_API/health" -Method Get -TimeoutSec 10
        Write-ColorOutput "✅ API is healthy: $($response.message)" "Green"
        return $true
    } catch {
        Write-ColorOutput "❌ API health check failed: $($_.Exception.Message)" "Red"
        return $false
    }
}

function Send-TestMessage {
    param(
        [string]$MessageContent,
        [string]$MessageType
    )
    
    Write-ColorOutput "`n📤 SENDING TEST MESSAGE" "Cyan"
    Write-ColorOutput ("=" * 50) "Cyan"
    
    # Generate unique message ID
    $messageId = "test_msg_" + [System.Guid]::NewGuid().ToString("N").Substring(0, 8)
    $timestamp = [DateTime]::Now.ToString("o")
    
    # Create message payload
    $messagePayload = @{
        message_id = $messageId
        chat_jid = "$SENDER_PHONE@s.whatsapp.net"
        chat_name = "Test User"
        sender_jid = "$SENDER_PHONE@s.whatsapp.net"
        sender_name = "Test User"
        user_identifier = $SENDER_PHONE
        content = $MessageContent
        content_type = "text"
        is_from_me = $false
        is_group = $false
        timestamp = $timestamp
        source_type = "whatsapp"
        priority = "normal"
    }
    
    Write-ColorOutput "📝 Message Type: $MessageType" "White"
    Write-ColorOutput "📞 From: $SENDER_PHONE" "White"
    Write-ColorOutput "📧 Message ID: $messageId" "White"
    
    $truncatedContent = if ($MessageContent.Length -gt 100) { 
        $MessageContent.Substring(0, 100) + "..." 
    } else { 
        $MessageContent 
    }
    Write-ColorOutput "💬 Content: $truncatedContent" "White"
    
    try {
        $jsonPayload = $messagePayload | ConvertTo-Json -Depth 3
        
        $response = Invoke-RestMethod -Uri "$PROCESSING_STORAGE_API/api/process-message" `
            -Method Post `
            -Body $jsonPayload `
            -ContentType "application/json" `
            -TimeoutSec 30
        
        Write-ColorOutput "`n✅ Message sent successfully!" "Green"
        Write-ColorOutput "📋 Stored ID: $($response.stored_id)" "White"
        Write-ColorOutput "🆔 Processing Job ID: $($response.processing_job_id)" "White"
        Write-ColorOutput "📊 Status: $($response.message)" "White"
        
        if ($response.validation_result) {
            Write-ColorOutput "✔️ Validation: $($response.validation_result | ConvertTo-Json -Compress)" "White"
        }
        
        return $response
    } catch {
        Write-ColorOutput "`n❌ Failed to send message: $($_.Exception.Message)" "Red"
        
        if ($_.Exception.Response) {
            $responseBody = $_.Exception.Response.GetResponseStream()
            $reader = [System.IO.StreamReader]::new($responseBody)
            $errorText = $reader.ReadToEnd()
            Write-ColorOutput "Error details: $errorText" "Red"
        }
        
        return $null
    }
}

function Get-RecentMessages {
    Write-ColorOutput "`n🔍 CHECKING RECENT MESSAGES" "Cyan"
    Write-ColorOutput ("=" * 50) "Cyan"
    
    try {
        $response = Invoke-RestMethod -Uri "$PROCESSING_STORAGE_API/api/records" -Method Get -TimeoutSec 10
        
        # Filter for messages from our test phone number
        $testRecords = $response | Where-Object { $_.user_identifier -eq $SENDER_PHONE }
        
        if ($testRecords) {
            Write-ColorOutput "📊 Found $($testRecords.Count) messages from $SENDER_PHONE" "Green"
            
            # Show the 3 most recent
            $recent = $testRecords | Select-Object -Last 3
            
            for ($i = 0; $i -lt $recent.Count; $i++) {
                $record = $recent[$i]
                Write-ColorOutput "`n📋 Message $($i + 1):" "Yellow"
                Write-ColorOutput "   🆔 ID: $($record.ID)" "White"
                Write-ColorOutput "   📧 Message ID: $($record.message_id)" "White"
                
                $truncatedContent = if ($record.content -and $record.content.Length -gt 80) {
                    $record.content.Substring(0, 80) + "..."
                } else {
                    $record.content
                }
                Write-ColorOutput "   💬 Content: $truncatedContent" "White"
                Write-ColorOutput "   📊 Status: $($record.processing_status)" "White"
                Write-ColorOutput "   ⏰ Created: $($record.created_at)" "White"
                
                if ($record.analysis_results) {
                    Write-ColorOutput "   🧠 Analysis: Available" "Green"
                }
                if ($record.extraction_results) {
                    Write-ColorOutput "   📄 Extraction: Available" "Green"
                }
            }
        } else {
            Write-ColorOutput "📭 No messages found from $SENDER_PHONE" "Yellow"
        }
    } catch {
        Write-ColorOutput "❌ Error checking messages: $($_.Exception.Message)" "Red"
    }
}

function Show-MessageTypes {
    Write-ColorOutput "Available message types:" "Cyan"
    foreach ($type in $TEST_MESSAGES.Keys) {
        $content = $TEST_MESSAGES[$type]
        $truncated = if ($content.Length -gt 60) { $content.Substring(0, 60) + "..." } else { $content }
        Write-ColorOutput "  $type`: $truncated" "White"
    }
}

# Main script logic
if ($Help) {
    Show-Help
    return
}

if ($ListTypes) {
    Show-MessageTypes
    return
}

Write-ColorOutput "🚀 WhatsApp Test Message Sender" "Cyan"
Write-ColorOutput ("=" * 50) "Cyan"

# Check API health first
if (-not (Test-ApiHealth)) {
    Write-ColorOutput "❌ Cannot proceed - API is not available" "Red"
    exit 1
}

if ($CheckOnly) {
    Get-RecentMessages
    return
}

# Determine message content
if ($CustomMessage) {
    $messageContent = $CustomMessage
    $messageTypeDisplay = "custom"
} elseif ($TEST_MESSAGES.ContainsKey($MessageType)) {
    $messageContent = $TEST_MESSAGES[$MessageType]
    $messageTypeDisplay = $MessageType
} else {
    Write-ColorOutput "❌ Invalid message type: $MessageType" "Red"
    Write-ColorOutput "Available types: $($TEST_MESSAGES.Keys -join ', ')" "Yellow"
    exit 1
}

# Send the message
$result = Send-TestMessage -MessageContent $messageContent -MessageType $messageTypeDisplay

if ($result) {
    Write-ColorOutput "`n⏳ Waiting 3 seconds before checking status..." "Yellow"
    Start-Sleep -Seconds 3
    
    # Check recent messages
    Get-RecentMessages
    
    Write-ColorOutput "`n✅ Test completed successfully!" "Green"
    Write-ColorOutput "💡 You can monitor processing in the service logs" "Yellow"
    Write-ColorOutput "📊 Check RabbitMQ management at: http://localhost:15672" "Yellow"
} else {
    Write-ColorOutput "`n❌ Test failed!" "Red"
    exit 1
}
