#!/usr/bin/env python3
"""
Send Test Message Script
Sends a test message to processing_storage_api.py as if it came from 212613880290
"""

import requests
import json
import uuid
from datetime import datetime
import sys
import argparse

# Configuration
PROCESSING_STORAGE_API = "http://localhost:8001"
SENDER_PHONE = "212613880290"

# Test messag es
TEST_MESSAGES = {
    "simple": "Hello, this is a test message from 212613880290",
    "fact_check": "BREAKING: Scientists at Harvard discovered that eating 5 bananas daily can completely reverse aging and add 50 years to your life. This simple fruit contains quantum proteins that repair DNA damage instantly!",
    "misinformation": "URGENT: The government is putting mind control chips in vaccines. Share this message to warn everyone before it's too late!",
    "covid": "COVID-19 vaccines contain 5G microchips that track your location and can control your thoughts remotely.",
    "custom": ""  # Will be set via command line
}

def check_api_health():
    """Check if the processing storage API is available"""
    print("🔍 Checking Processing Storage API health...")
    try:
        response = requests.get(f"{PROCESSING_STORAGE_API}/health", timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ API is healthy: {result.get('message', 'OK')}")
            return True
        else:
            print(f"❌ API returned HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API health check failed: {e}")
        return False

def check_rabbitmq_status():
    """Check if RabbitMQ is accessible"""
    print("🔍 Checking RabbitMQ status...")
    try:
        # Check RabbitMQ management interface
        response = requests.get("http://localhost:15672", timeout=5)
        if response.status_code == 200:
            print("✅ RabbitMQ management interface is accessible")
            return True
        else:
            print(f"⚠️ RabbitMQ management returned HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ RabbitMQ check failed: {e}")
        print("💡 Make sure RabbitMQ Docker container is running:")
        print("   docker run -d --name rabbitmq_server -p 5672:5672 -p 15672:15672 rabbitmq:3-management")
        return False

def send_test_message(message_content, message_type="custom"):
    """Send a test message to the processing storage API"""
    print(f"\n📤 SENDING TEST MESSAGE")
    print("=" * 50)
    
    # Generate unique message ID
    message_id = f"test_msg_{uuid.uuid4().hex[:8]}"
    
    # Create message payload
    message_payload = {
        "message_id": message_id,
        "chat_jid": f"{SENDER_PHONE}@s.whatsapp.net",
        "chat_name": "Test User",
        "sender_jid": f"{SENDER_PHONE}@s.whatsapp.net", 
        "sender_name": "Test User",
        "user_identifier": SENDER_PHONE,
        "content": message_content,
        "content_type": "text",
        "is_from_me": False,
        "is_group": False,
        "timestamp": datetime.now().isoformat(),
        "source_type": "whatsapp",
        "priority": "normal"
    }
    
    print(f"📝 Message Type: {message_type}")
    print(f"📞 From: {SENDER_PHONE}")
    print(f"📧 Message ID: {message_id}")
    print(f"💬 Content: {message_content[:100]}{'...' if len(message_content) > 100 else ''}")
    
    try:
        response = requests.post(
            f"{PROCESSING_STORAGE_API}/api/process-message",
            json=message_payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Message sent successfully!")
            print(f"📋 Stored ID: {result.get('stored_id', 'N/A')}")
            print(f"🆔 Processing Job ID: {result.get('processing_job_id', 'N/A')}")
            print(f"📊 Status: {result.get('message', 'N/A')}")
            
            # Check for processing issues
            if result.get('processing_job_id') is None:
                print(f"⚠️ Warning: No processing job ID - check RabbitMQ status")
                print(f"📋 Full response: {json.dumps(result, indent=2)}")
            
            if result.get('validation_result'):
                print(f"✔️ Validation: {result['validation_result']}")
            
            return result
        else:
            print(f"\n❌ Failed to send message: HTTP {response.status_code}")
            try:
                error_detail = response.json()
                print(f"Error details: {json.dumps(error_detail, indent=2)}")
            except:
                print(f"Response text: {response.text}")
            return None
            
    except Exception as e:
        print(f"\n❌ Error sending message: {e}")
        return None

def check_recent_messages():
    """Check recent messages in the database"""
    print(f"\n🔍 CHECKING RECENT MESSAGES")
    print("=" * 50)
    
    try:
        response = requests.get(f"{PROCESSING_STORAGE_API}/api/records", timeout=10)
        if response.status_code == 200:
            data = response.json()
            
            # Debug: Check the structure of the response
            print(f"📊 Response structure: {type(data)}")
            
            # Handle different response structures
            if isinstance(data, dict):
                if 'data' in data:
                    records = data['data']
                    print(f"📊 Found {len(records)} records (from data key)")
                elif 'records' in data:
                    records = data['records']
                    print(f"📊 Found {len(records)} records (from records key)")
                else:
                    print(f"📊 Dict keys: {list(data.keys())}")
                    records = []
            elif isinstance(data, list):
                records = data
                print(f"📊 Found {len(records)} records (direct list)")
            else:
                print(f"❌ Unexpected response format: {type(data)}")
                return
            
            # Filter for messages from our test phone number
            test_records = []
            for r in records:
                if isinstance(r, dict) and r.get('user_identifier') == SENDER_PHONE:
                    test_records.append(r)
            
            if test_records:
                print(f"📊 Found {len(test_records)} messages from {SENDER_PHONE}")
                
                # Show the 3 most recent
                recent = test_records[-3:] if len(test_records) >= 3 else test_records
                
                for i, record in enumerate(recent, 1):
                    print(f"\n📋 Message {i}:")
                    print(f"   🆔 ID: {record.get('ID', 'N/A')}")
                    print(f"   📧 Message ID: {record.get('message_id', 'N/A')}")
                    print(f"   💬 Content: {record.get('content', 'N/A')[:80]}...")
                    print(f"   📊 Status: {record.get('processing_status', 'N/A')}")
                    print(f"   ⏰ Created: {record.get('created_at', 'N/A')}")
                    
                    if record.get('analysis_results'):
                        print(f"   🧠 Analysis: Available")
                    if record.get('extraction_results'):
                        print(f"   📄 Extraction: Available")
            else:
                print(f"📭 No messages found from {SENDER_PHONE}")
                
        else:
            print(f"❌ Failed to get records: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error checking messages: {e}")

def main():
    parser = argparse.ArgumentParser(description="Send test message to WhatsApp processing API")
    parser.add_argument("--type", choices=list(TEST_MESSAGES.keys()), default="simple",
                      help="Type of test message to send")
    parser.add_argument("--message", type=str, help="Custom message content")
    parser.add_argument("--check-only", action="store_true", help="Only check recent messages, don't send")
    parser.add_argument("--list-types", action="store_true", help="List available message types")
    
    args = parser.parse_args()
    
    if args.list_types:
        print("Available message types:")
        for msg_type, content in TEST_MESSAGES.items():
            if msg_type != "custom":
                print(f"  {msg_type}: {content[:60]}...")
        return
    
    print("🚀 WhatsApp Test Message Sender")
    print("=" * 50)
    
    # Check API health first
    if not check_api_health():
        print("❌ Cannot proceed - API is not available")
        sys.exit(1)
    
    # Check RabbitMQ status (optional but helpful)
    rabbitmq_ok = check_rabbitmq_status()
    if not rabbitmq_ok:
        print("⚠️ RabbitMQ may not be available - messages will be stored but not processed")
        user_input = input("Continue anyway? (y/N): ")
        if user_input.lower() != 'y':
            sys.exit(1)
    
    if args.check_only:
        check_recent_messages()
        return
    
    # Determine message content
    if args.message:
        message_content = args.message
        message_type = "custom"
    elif args.type == "custom":
        print("Please provide a custom message with --message flag")
        sys.exit(1)
    else:
        message_content = TEST_MESSAGES[args.type]
        message_type = args.type
    
    # Send the message
    result = send_test_message(message_content, message_type)
    
    if result:
        print(f"\n⏳ Waiting 3 seconds before checking status...")
        import time
        time.sleep(3)
        
        # Check recent messages
        check_recent_messages()
        
        print(f"\n✅ Test completed successfully!")
        print(f"💡 You can monitor processing in the service logs")
        print(f"📊 Check RabbitMQ management at: http://localhost:15672")
    else:
        print(f"\n❌ Test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
