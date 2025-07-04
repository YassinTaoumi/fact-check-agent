#!/usr/bin/env python3
"""
RabbitMQ Connection Reset Utility

This script helps troubleshoot and fix RabbitMQ connection issues by:
1. Checking RabbitMQ connection status
2. Attempting to close stale connections
3. Providing guidance for common issues

Usage:
  python reset_rabbitmq_connections.py
"""

import os
import sys
import socket
import time
import requests
import pika
import subprocess
from typing import Tuple, Dict, List, Optional, Any

def check_rabbitmq_server() -> Tuple[bool, Optional[str]]:
    """Check if RabbitMQ server is accessible via socket"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        
        # Default RabbitMQ port
        result = s.connect_ex(('localhost', 5672))
        s.close()
        
        if result == 0:
            return True, None
        else:
            return False, f"RabbitMQ port 5672 is not open (error code: {result})"
    except Exception as e:
        return False, f"Failed to check RabbitMQ: {str(e)}"

def try_rabbitmq_connection() -> Tuple[bool, Optional[str]]:
    """Try to establish a RabbitMQ connection"""
    try:
        # Get connection parameters from environment or use defaults
        host = os.getenv("RABBITMQ_HOST", "localhost")
        port = int(os.getenv("RABBITMQ_PORT", "5672"))
        user = os.getenv("RABBITMQ_USER", "guest")
        password = os.getenv("RABBITMQ_PASSWORD", "guest")
        
        # Create connection parameters
        credentials = pika.PlainCredentials(user, password)
        parameters = pika.ConnectionParameters(
            host=host,
            port=port,
            credentials=credentials,
            connection_attempts=2,
            retry_delay=1
        )
        
        # Try to establish connection
        connection = pika.BlockingConnection(parameters)
        
        # If we get here, connection was successful
        print("Successfully connected to RabbitMQ")
        
        # Create a channel
        channel = connection.channel()
        
        # Check a few queues to ensure system is working
        try:
            # Declare and check test queue
            channel.queue_declare(queue='test_queue', durable=True)
            print("Successfully created test queue")
            
            # Verify standard queues existence
            print("Checking standard queues...")
            result = channel.queue_declare(queue='fact_check_queue', durable=True, passive=True)
            print(f"fact_check_queue exists with {result.method.message_count} messages")
        except Exception as e:
            print(f"⚠️ Queue check error: {e}")
        
        # Close the connection properly
        connection.close()
        print("Connection closed properly")
        
        return True, None
    
    except pika.exceptions.AMQPConnectionError as e:
        return False, f"AMQP Connection Error: {e}"
    except pika.exceptions.AuthenticationError as e:
        return False, f"Authentication Error: {e}"
    except Exception as e:
        return False, f"Unexpected error connecting to RabbitMQ: {e}"

def reset_rabbitmq_connections() -> bool:
    """
    Attempt to reset RabbitMQ connections using management API
    Requires management plugin to be enabled
    """
    try:
        # Try to use management API (default credentials)
        username = os.getenv("RABBITMQ_USER", "guest")
        password = os.getenv("RABBITMQ_PASSWORD", "guest") 
        
        # Get all connections
        response = requests.get(
            "http://localhost:15672/api/connections",
            auth=(username, password)
        )
        
        if response.status_code != 200:
            print(f"⚠️ Failed to access RabbitMQ management API: HTTP {response.status_code}")
            return False
        
        connections = response.json()
        print(f"Found {len(connections)} active RabbitMQ connections")
        
        # Close each connection
        closed_count = 0
        for conn in connections:
            conn_name = conn.get('name', 'unknown')
            print(f"Closing connection: {conn_name}")
            
            close_response = requests.delete(
                f"http://localhost:15672/api/connections/{conn_name}",
                auth=(username, password)
            )
            
            if close_response.status_code == 204:
                closed_count += 1
                print(f"Successfully closed connection: {conn_name}")
            else:
                print(f"Failed to close connection {conn_name}: HTTP {close_response.status_code}")
        
        print(f"Closed {closed_count} out of {len(connections)} connections")
        return closed_count > 0
    
    except requests.exceptions.ConnectionError:
        print("⚠️ RabbitMQ Management API not accessible (http://localhost:15672)")
        return False
    except Exception as e:
        print(f"⚠️ Error resetting connections: {e}")
        return False

def suggest_restart_commands():
    """Display commands to restart RabbitMQ based on platform"""
    print("\nCommands to restart RabbitMQ:")
    
    if sys.platform.startswith('win'):
        print("🪟 Windows:")
        print("  1. Open Services (services.msc)")
        print("  2. Find 'RabbitMQ' service")
        print("  3. Right-click and select 'Restart'")
        print("  OR use PowerShell (as Administrator):")
        print("  > Restart-Service RabbitMQ")
    
    elif sys.platform.startswith('linux'):
        print("🐧 Linux:")
        print("  > sudo systemctl restart rabbitmq-server")
    
    elif sys.platform.startswith('darwin'):
        print("🍎 macOS:")
        print("  > brew services restart rabbitmq")
        print("  OR")
        print("  > rabbitmqctl stop_app && rabbitmqctl start_app")
    
    print("\n💡 After restarting, wait 10-15 seconds for RabbitMQ to initialize")

def main():
    print("RabbitMQ Connection Reset Utility")
    print("=" * 50)
    
    # Step 1: Check if server is running
    print("\nStep 1: Checking RabbitMQ server status...")
    server_running, server_error = check_rabbitmq_server()
    
    if not server_running:
        print(f"❌ RabbitMQ server issue: {server_error}")
        print("\n⚠️ RabbitMQ server doesn't appear to be running!")
        suggest_restart_commands()
        return
    
    print("RabbitMQ server is running (port 5672 is open)")
    
    # Step 2: Check connection
    print("\nStep 2: Testing RabbitMQ connection...")
    connection_ok, connection_error = try_rabbitmq_connection()
    
    if not connection_ok:
        print(f"❌ Connection test failed: {connection_error}")
        
        # Step 3: Try to reset connections
        print("\nStep 3: Attempting to reset stale connections...")
        reset_success = reset_rabbitmq_connections()
        
        if reset_success:
            print("\nSuccessfully reset connections")
            print("Waiting 5 seconds for RabbitMQ to process changes...")
            time.sleep(5)
            
            # Try connection again
            print("\nTesting connection again...")
            retry_ok, retry_error = try_rabbitmq_connection()
            
            if retry_ok:
                print("\nConnection is now working!")
            else:
                print(f"\nConnection still failing: {retry_error}")
                suggest_restart_commands()
        else:
            print("\nFailed to reset connections")
            suggest_restart_commands()
    else:
        print("\nRabbitMQ connection is working normally!")
        print("Your fact-checking pipeline should work now")
    
    print("\nTroubleshooting Tips:")
    print("1. Make sure all services are running:")
    print("   - RabbitMQ Server")
    print("   - processing_storage_api.py")
    print("   - main_combined.py")
    print("   - RabbitMQ worker scripts")
    print("2. Check for firewall or antivirus software blocking connections")
    print("3. Look for error messages in the logs of your services")
    print("4. Try restarting all components if problems persist")

if __name__ == "__main__":
    main()
