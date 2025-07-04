"""
Setup and installation script for the fact-checker system.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print(f"Error: requirements.txt not found at {requirements_file}")
        return False
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False

def setup_environment():
    """Set up environment file."""
    print("Setting up environment configuration...")
    
    env_example = Path(__file__).parent / ".env.example"
    env_file = Path(__file__).parent / ".env"
    
    if env_file.exists():
        print("✓ .env file already exists")
        return True
    
    if not env_example.exists():
        print("✗ .env.example file not found")
        return False
    
    # Copy example to .env
    try:
        with open(env_example, 'r') as f:
            content = f.read()
        
        with open(env_file, 'w') as f:
            f.write(content)
        
        print("✓ Created .env file from template")
        print("⚠ Please edit .env file and add your API keys")
        return True
    except Exception as e:
        print(f"✗ Error creating .env file: {e}")
        return False

def verify_configuration():
    """Verify system configuration."""
    print("Verifying configuration...")
    
    try:
        from config import config
        
        # Check required settings
        checks = [
            ("Google AI API Key", bool(config.google_ai_api_key)),
            ("SearXNG URL", bool(config.searxng_url)),
            ("Model Name", bool(config.model_name)),
        ]
        
        all_good = True
        for check_name, check_result in checks:
            status = "✓" if check_result else "✗"
            print(f"  {status} {check_name}")
            if not check_result:
                all_good = False
        
        if all_good:
            print("✓ Configuration verified")
        else:
            print("⚠ Some configuration issues found - please check .env file")
        
        return all_good
        
    except Exception as e:
        print(f"✗ Error verifying configuration: {e}")
        return False

def test_components():
    """Test system components."""
    print("Testing system components...")
    
    try:
        # Test imports
        from claim_extractor import ClaimExtractor
        from web_searcher import WebSearcher
        from content_crawler import ContentCrawler
        from summarizer import ContentSummarizer
        from verdict_analyzer import VerdictAnalyzer
        from fact_checker import FactChecker
        
        print("✓ All components imported successfully")
        
        # Test basic initialization
        checker = FactChecker()
        workflow_info = checker.get_workflow_info()
        
        print(f"✓ Fact-checker initialized with {len(workflow_info['workflow_steps'])} steps")
        print(f"✓ Using model: {workflow_info['configuration']['model']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing components: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("Creating directories...")
    
    directories = [
        "logs",
        "results",
        "cache"
    ]
    
    base_path = Path(__file__).parent
    
    for directory in directories:
        dir_path = base_path / directory
        try:
            dir_path.mkdir(exist_ok=True)
            print(f"✓ Created directory: {directory}")
        except Exception as e:
            print(f"✗ Error creating directory {directory}: {e}")

def show_usage_info():
    """Show usage information."""
    print("\nSETUP COMPLETE!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Edit the .env file and add your API keys:")
    print("   - GOOGLE_AI_API_KEY: Your Google AI Studio API key")
    print("   - SEARXNG_URL: URL of your SearXNG instance")
    print("\n2. Start a SearXNG instance (if not already running):")
    print("   docker run -d -p 8080:8080 searxng/searxng")
    print("\n3. Test the system:")
    print("   python examples.py")
    print("\n4. Use the CLI:")
    print("   python cli.py check \"Your text to fact-check\"")
    print("   python cli.py interactive")
    print("\n5. Use in Python:")
    print("   from fact_checker import FactChecker")
    print("   checker = FactChecker()")
    print("   results = await checker.check_facts('Your text')")

def main():
    """Main setup function."""
    print("FACT-CHECKER SYSTEM SETUP")
    print("=" * 50)
    
    steps = [
        ("Installing requirements", install_requirements),
        ("Setting up environment", setup_environment),
        ("Creating directories", create_directories),
        ("Verifying configuration", verify_configuration),
        ("Testing components", test_components),
    ]
    
    success_count = 0
    
    for step_name, step_function in steps:
        print(f"\n{step_name}...")
        if step_function():
            success_count += 1
        else:
            print(f"⚠ Step '{step_name}' had issues")
    
    print(f"\nSetup completed: {success_count}/{len(steps)} steps successful")
    
    if success_count == len(steps):
        show_usage_info()
    else:
        print("\n⚠ Some setup steps failed. Please check the errors above.")
        print("You may need to manually install dependencies or configure settings.")

if __name__ == "__main__":
    main()
