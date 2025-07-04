"""
WhatsApp Bridge Services Launcher

This script starts all required services for the WhatsApp Bridge system:
- Main Combined API (port 8000)
- Processing Storage API (port 8001)
- Go WhatsApp Client
- RabbitMQ Orchestrator
- RabbitMQ Workers for different content types

Each service runs in a separate process to handle concurrent processing.
"""

import subprocess
import sys
import time
import signal
import os
from multiprocessing import Process
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service configurations
SERVICES = {
    'main_combined': {
        'script': 'main_combined.py',
        'args': [],
        'description': 'Main Combined API (port 8000) - Primary API with RabbitMQ integration',
        'type': 'python'
    },
    'processing_storage': {
        'script': 'processing_storage_api.py',
        'args': [],
        'description': 'Processing Storage API (port 8001) - File processing and storage',
        'type': 'python'
    },
    'go_client': {
        'script': 'main.go',
        'args': [],
        'description': 'Go WhatsApp Client - Handles WhatsApp communication',
        'type': 'go'
    },
    'rabbitmq_orchestrator': {
        'script': 'rabbitmq_orchestrator.py',
        'args': [],
        'description': 'RabbitMQ Orchestrator - Message routing and queue management',
        'type': 'python'
    },
    'image_worker': {
        'script': 'rabbitmq_workers.py',
        'args': ['image'],
        'description': 'Image Processing Worker (OCR, AI Detection, Modification Detection)',
        'type': 'python'
    },
    'video_worker': {
        'script': 'rabbitmq_workers.py', 
        'args': ['video'],
        'description': 'Video/Audio Processing Worker (Transcription)',
        'type': 'python'
    },
    'pdf_worker': {
        'script': 'rabbitmq_workers.py',
        'args': ['pdf'], 
        'description': 'PDF Processing Worker (Text Extraction)',
        'type': 'python'
    },
    'text_worker': {
        'script': 'rabbitmq_workers.py',
        'args': ['text'],
        'description': 'Text Processing Worker (Link Crawling, Fact Checking)',
        'type': 'python'
    }
}

class ServiceManager:
    """Manages multiple WhatsApp Bridge services"""
    
    def __init__(self):
        self.processes = {}
        self.running = False
    
    def start_service(self, service_name: str, config: dict):
        """Start a single service process"""
        try:
            logger.info(f"Starting {service_name} service: {config['description']}")
            
            # Determine command based on service type
            if config['type'] == 'go':
                # For Go services, use 'go run'
                cmd = ['go', 'run', config['script']] + config['args']
            else:
                # For Python services
                cmd = [sys.executable, config['script']] + config['args']
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.getcwd()
            )
            
            self.processes[service_name] = {
                'process': process,
                'config': config
            }
            
            logger.info(f"{service_name} service started with PID: {process.pid}")
            
        except Exception as e:
            logger.error(f"Failed to start {service_name} service: {e}")
    
    def start_all_services(self):
        """Start all configured services"""
        logger.info("Starting all WhatsApp Bridge services...")
        
        for service_name, config in SERVICES.items():
            self.start_service(service_name, config)
            time.sleep(2)  # Small delay between starts to avoid port conflicts
        
        self.running = True
        logger.info(f"All {len(SERVICES)} services started successfully")
    
    def stop_service(self, service_name: str):
        """Stop a single service"""
        if service_name in self.processes:
            process_info = self.processes[service_name]
            process = process_info['process']
            
            logger.info(f"Stopping {service_name} service (PID: {process.pid})")
            
            try:
                process.terminate()
                process.wait(timeout=10)
                logger.info(f"{service_name} service stopped gracefully")
            except subprocess.TimeoutExpired:
                logger.warning(f"Force killing {service_name} service")
                process.kill()
                process.wait()
            except Exception as e:
                logger.error(f"Error stopping {service_name}: {e}")
            
            del self.processes[service_name]
    
    def stop_all_services(self):
        """Stop all services"""
        logger.info("Stopping all services...")
        
        for service_name in list(self.processes.keys()):
            self.stop_service(service_name)
        
        self.running = False
        logger.info("All services stopped")
    
    def check_services(self):
        """Check status of all services and restart if needed"""
        for service_name, process_info in list(self.processes.items()):
            process = process_info['process']
            
            if process.poll() is not None:
                # Process has died
                logger.warning(f"{service_name} service died, restarting...")
                del self.processes[service_name]
                self.start_service(service_name, process_info['config'])
    
    def run(self):
        """Main run loop"""
        self.start_all_services()
        
        try:
            while self.running:
                time.sleep(5)
                self.check_services()
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
        finally:
            self.stop_all_services()

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)

if __name__ == "__main__":
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start service manager
    manager = ServiceManager()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--single':
        # Run single service for testing
        if len(sys.argv) < 3:
            print("Usage: python start_workers.py --single <service_name>")
            print(f"Available services: {', '.join(SERVICES.keys())}")
            sys.exit(1)
        
        service_name = sys.argv[2]
        if service_name not in SERVICES:
            print(f"Invalid service name: {service_name}")
            print(f"Available services: {', '.join(SERVICES.keys())}")
            sys.exit(1)
        
        manager.start_service(service_name, SERVICES[service_name])
        
        try:
            while True:
                time.sleep(1)
                manager.check_services()
        except KeyboardInterrupt:
            manager.stop_service(service_name)
    else:
        # Run all services
        print("Starting WhatsApp Bridge System...")
        print("This will start:")
        for service_name, config in SERVICES.items():
            print(f"  - {service_name}: {config['description']}")
        print()
        manager.run()
