#!/usr/bin/env python3
"""
Quick Start Script for MT5 Trading Bot Web Interface
====================================================

This script starts the web interface for the MT5 trading bot.
"""

import os
import socket
import subprocess
import sys

def get_local_ip():
    """Get the local IP address of the computer"""
    try:
        # Connect to a remote address to get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "localhost"

def check_dependencies():
    """Check and install required dependencies"""
    required_packages = ['flask', 'python-dotenv']
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} not found. Installing...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ {package} installed successfully")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")
                return False
    return True

def main():
    print("🌐 MT5 Trading Bot - Web Interface")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Failed to install required dependencies")
        return
    
    # Get local IP
    local_ip = get_local_ip()
    
    print(f"\n📱 Access from your phone:")
    print(f"   http://{local_ip}:5000")
    print(f"\n💻 Access from your PC:")
    print(f"   http://localhost:5000")
    print(f"\n🔧 Make sure your phone and PC are on the same WiFi network!")
    print(f"\n⚠️  IMPORTANT: Make sure MT5 is running and AutoTrading is enabled!")
    print(f"\n🚀 Starting web interface...")
    print("=" * 50)
    
    # Start the web interface
    try:
        from web_interface import app, create_templates
        create_templates()  # Ensure template exists before Flask starts
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n🛑 Web interface stopped by user")
    except Exception as e:
        print(f"❌ Error starting web interface: {e}")
        print("\n💡 Make sure you're in the mt5-trading directory and all files are present")

if __name__ == "__main__":
    main() 