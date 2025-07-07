#!/usr/bin/env python3
"""
Unified MT5 Trading Bot Web Interface - Start Script
====================================================

This script starts the unified web interface for the MT5 trading bot.
Optimized for small accounts ($10+) with proper scaling strategies.
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
    required_packages = ['flask', 'python-dotenv', 'MetaTrader5', 'pandas', 'numpy']
    
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

def check_mt5_installation():
    """Check if MetaTrader5 is properly installed"""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            print("❌ MT5 initialization failed. Make sure MetaTrader5 is installed and running.")
            return False
        mt5.shutdown()
        print("✅ MetaTrader5 is properly installed")
        return True
    except ImportError:
        print("❌ MetaTrader5 package not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "MetaTrader5"])
            print("✅ MetaTrader5 installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install MetaTrader5")
            return False

def check_config_files():
    """Check if required configuration files exist"""
    config_files = [
        'real-trading/real.env',
        'real-trading/real_config.py',
        'real-trading/micro_scaling_config.py'
    ]
    
    missing_files = []
    for file_path in config_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing configuration files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("✅ All configuration files found")
    return True

def main():
    print("🌐 MT5 Trading Bot - Unified Web Interface")
    print("=" * 60)
    print("💰 Optimized for small account scaling ($10+)")
    print("📈 Progressive volume scaling strategy")
    print("🎯 Target: Scale from $10 to $100+ systematically")
    print("=" * 60)
    
    # Check dependencies
    print("\n🔧 Checking dependencies...")
    if not check_dependencies():
        print("❌ Failed to install required dependencies")
        return
    
    # Check MT5 installation
    print("\n📊 Checking MetaTrader5 installation...")
    if not check_mt5_installation():
        print("❌ MetaTrader5 installation issues")
        return
    
    # Check configuration files
    print("\n📁 Checking configuration files...")
    if not check_config_files():
        print("❌ Configuration files missing")
        return
    
    # Get local IP
    local_ip = get_local_ip()
    
    print(f"\n📱 Access from your phone:")
    print(f"   http://{local_ip}:5000")
    print(f"\n💻 Access from your PC:")
    print(f"   http://localhost:5000")
    print(f"\n🔧 Make sure your phone and PC are on the same WiFi network!")
    print(f"\n⚠️  IMPORTANT: Make sure MT5 is running and AutoTrading is enabled!")
    print(f"\n💰 ACCOUNT INFO:")
    print(f"   - Real Account: $10+ balance")
    print(f"   - Trading: Volatility 100 Index")
    print(f"   - Strategy: Progressive scaling")
    print(f"   - Target: Scale to $100+ systematically")
    print(f"\n🚀 Starting unified web interface...")
    print("=" * 60)
    
    # Start the unified web interface
    try:
        from unified_web_interface import app, create_unified_template
        create_unified_template()  # Ensure template exists before Flask starts
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n🛑 Web interface stopped by user")
    except Exception as e:
        print(f"❌ Error starting web interface: {e}")
        print("\n💡 Make sure you're in the mt5-trading directory and all files are present")

if __name__ == "__main__":
    main() 