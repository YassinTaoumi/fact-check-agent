#!/usr/bin/env python3
"""
Diagnostic Script for WhatsApp Pipeline
Checks the status of all services and APIs
"""

import requests
import json
import sys

# Configuration
PROCESSING_STORAGE_API = "http://localhost:8001"
MAIN_COMBINED_API = "http://localhost:8000"
WHATSAPP_API = "http://localhost:9090"
RABBITMQ_MANAGEMENT = "http://localhost:15672"

def check_endpoint(name, url, expected_paths=None):
    """Check if an endpoint is accessible and optionally test specific paths"""
    print(f"\n🔍 CHECKING {name}")
    print("-" * 50)
    
    base_working = False
    
    # Check base URL
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            print(f"✅ {url} - ACCESSIBLE")
            base_working = True
        else:
            print(f"⚠️ {url} - HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ {url} - {e}")
    
    # Check specific paths if provided
    if expected_paths and base_working:
        for path in expected_paths:
            full_url = f"{url.rstrip('/')}/{path.lstrip('/')}"
            try:
                response = requests.get(full_url, timeout=5)
                if response.status_code == 200:
                    print(f"✅ {path} - WORKING")
                    # Show response for health endpoints
                    if 'health' in path:
                        try:
                            result = response.json()
                            print(f"   📊 Response: {json.dumps(result, indent=6)}")
                        except:
                            print(f"   📊 Response: {response.text[:200]}")
                else:
                    print(f"⚠️ {path} - HTTP {response.status_code}")
            except Exception as e:
                print(f"❌ {path} - {e}")
    
    return base_working

def check_database_records():
    """Check what's in the database"""
    print(f"\n📊 CHECKING DATABASE RECORDS")
    print("-" * 50)
    
    try:
        response = requests.get(f"{PROCESSING_STORAGE_API}/api/records", timeout=10)
        print(f"Records endpoint status: HTTP {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response type: {type(data)}")
            
            if isinstance(data, dict):
                print(f"Response keys: {list(data.keys())}")
                if 'data' in data:
                    records = data['data']
                elif 'records' in data:
                    records = data['records']
                else:
                    print("Raw response:")
                    print(json.dumps(data, indent=2)[:500])
                    return
            elif isinstance(data, list):
                records = data
            else:
                print(f"Unexpected data type: {type(data)}")
                return
            
            print(f"Found {len(records)} total records")
            
            # Show recent records
            if records:
                recent = records[-3:] if len(records) >= 3 else records
                for i, record in enumerate(recent, 1):
                    if isinstance(record, dict):
                        print(f"\nRecord {i}:")
                        print(f"  ID: {record.get('ID', 'N/A')}")
                        print(f"  Message ID: {record.get('message_id', 'N/A')}")
                        print(f"  User: {record.get('user_identifier', 'N/A')}")
                        print(f"  Content: {str(record.get('content', 'N/A'))[:50]}...")
                        print(f"  Status: {record.get('processing_status', 'N/A')}")
                    else:
                        print(f"Record {i}: {type(record)} - {record}")
        else:
            print(f"Failed to get records: {response.text}")
            
    except Exception as e:
        print(f"Error checking records: {e}")

def check_rabbitmq_details():
    """Check RabbitMQ in detail"""
    print(f"\n🐰 CHECKING RABBITMQ DETAILS")
    print("-" * 50)
    
    try:
        # Check overview
        response = requests.get(f"{RABBITMQ_MANAGEMENT}/api/overview", 
                               auth=("guest", "guest"), timeout=10)
        if response.status_code == 200:
            overview = response.json()
            print(f"✅ RabbitMQ API accessible")
            print(f"   Version: {overview.get('rabbitmq_version', 'N/A')}")
            print(f"   Node: {overview.get('node', 'N/A')}")
        
        # Check queues
        response = requests.get(f"{RABBITMQ_MANAGEMENT}/api/queues", 
                               auth=("guest", "guest"), timeout=10)
        if response.status_code == 200:
            queues = response.json()
            print(f"   Queues: {len(queues)} found")
            for queue in queues:
                name = queue.get('name', 'unnamed')
                messages = queue.get('messages', 0)
                print(f"     - {name}: {messages} messages")
        
    except Exception as e:
        print(f"❌ RabbitMQ details failed: {e}")

def main():
    print("🔧 WHATSAPP PIPELINE DIAGNOSTIC")
    print("=" * 60)
    
    # Check all services
    services = [
        ("Processing Storage API", PROCESSING_STORAGE_API, ["health", "api/records"]),
        ("Main Combined API", MAIN_COMBINED_API, ["api/health"]),
        ("WhatsApp API", WHATSAPP_API, ["health"]),
        ("RabbitMQ Management", RABBITMQ_MANAGEMENT, [])
    ]
    
    working_services = 0
    for name, url, paths in services:
        if check_endpoint(name, url, paths):
            working_services += 1
    
    print(f"\n📈 SUMMARY")
    print("-" * 50)
    print(f"Working services: {working_services}/{len(services)}")
    
    # Detailed checks
    check_database_records()
    check_rabbitmq_details()
    
    print(f"\n✅ Diagnostic complete!")

if __name__ == "__main__":
    main()
