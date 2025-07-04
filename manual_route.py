#!/usr/bin/env python3
"""
Script to manually route a stored message to RabbitMQ queue
"""
from rabbitmq_orchestrator import RabbitMQOrchestrator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def manual_route_message(message_id: str, queue_name: str):
    """Manually route a message to RabbitMQ queue"""
    try:
        # Initialize orchestrator
        orchestrator = RabbitMQOrchestrator()
        
        print(f"🔄 Routing message {message_id} to {queue_name}")
        
        # Route the message
        orchestrator.route_to_queue(message_id, queue_name)
        
        print(f"✅ Successfully routed message {message_id} to {queue_name}")
        
        # Close connections
        orchestrator.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to route message: {e}")
        return False

if __name__ == "__main__":
    # Route the test image
    message_id = "5a26a07004c541a6"
    queue_name = "image_processing_queue"
    
    success = manual_route_message(message_id, queue_name)
    
    if success:
        print("\n🎉 Manual routing completed successfully!")
        print("💡 Check the image processing worker logs to see if it processes the image")
    else:
        print("\n❌ Manual routing failed!")
