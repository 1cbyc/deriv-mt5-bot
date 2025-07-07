#!/usr/bin/env python3
"""
Unified Trading Bot Setup Script
================================

This script helps you set up the trading bot system.
"""

import os
import sys
import subprocess
from pathlib import Path

def print_banner():
    """Print the setup banner"""
    print("🔧 Unified Trading Bot Setup")
    print("=" * 40)
    print("📊 Complete Trading Solution Setup")
    print("💰 Optimized for small account scaling")
    print("=" * 40)

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ is required")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} is compatible")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        # Install from requirements.txt
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_mt5_installation():
    """Check and install MetaTrader5 if needed"""
    print("\n📊 Checking MetaTrader5...")
    
    try:
        import MetaTrader5 as mt5
        if mt5.initialize():
            mt5.shutdown()
            print("✅ MetaTrader5 is properly installed")
            return True
        else:
            print("⚠️  MT5 initialization failed")
            return False
    except ImportError:
        print("📦 Installing MetaTrader5...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "MetaTrader5"])
            print("✅ MetaTrader5 installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install MetaTrader5")
            return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        "logs",
        "data",
        "backtests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created {directory}/")

def create_example_configs():
    """Create example configuration files"""
    print("\n📝 Creating example configurations...")
    
    # Deriv trading .env example
    deriv_env_content = """# Deriv Trading Configuration
# ===========================

# Deriv API Configuration
DERIV_APP_ID=1089
DERIV_ACCOUNT_TOKEN=your_token_here
DERIV_IS_DEMO=true

# Trading Configuration
DERIV_DEFAULT_AMOUNT=1.00
DERIV_CURRENCY=USD

# Risk Management
DERIV_MAX_DAILY_LOSS=50.00
DERIV_MAX_DAILY_TRADES=20

# Instructions:
# 1. Replace 'your_token_here' with your actual Deriv account token
# 2. Set DERIV_IS_DEMO=false for real trading
# 3. Adjust amounts and risk settings as needed
"""
    
    deriv_dir = Path("deriv_trading")
    if deriv_dir.exists():
        env_file = deriv_dir / ".env.example"
        try:
            with open(env_file, 'w') as f:
                f.write(deriv_env_content)
            print("✅ Created deriv_trading/.env.example")
        except Exception as e:
            print(f"❌ Error creating .env.example: {e}")

def check_mt5_config():
    """Check MT5 configuration files"""
    print("\n🎮 Checking MT5 configuration...")
    
    mt5_dir = Path("mt5-trading")
    if not mt5_dir.exists():
        print("❌ MT5 trading directory not found")
        return False
    
    # Check for real trading config
    real_env = mt5_dir / "real-trading" / "real.env"
    if not real_env.exists():
        print("⚠️  Real trading configuration not found")
        print("📝 You'll need to create mt5-trading/real-trading/real.env")
        print("   with your MT5 credentials when ready for real trading")
    else:
        print("✅ Real trading configuration found")
    
    # Check for demo trading
    demo_dir = mt5_dir / "demo-trading"
    if demo_dir.exists():
        print("✅ Demo trading directory found")
    else:
        print("❌ Demo trading directory not found")
    
    return True

def show_next_steps():
    """Show next steps for the user"""
    print("\n🎯 Next Steps:")
    print("1. 📊 For MT5 Trading:")
    print("   - Install MetaTrader 5 terminal")
    print("   - Enable AutoTrading in MT5")
    print("   - Run: python start_trading.py")
    print("   - Choose option 1 (MT5 Trading)")
    print()
    print("2. 🌐 For Deriv Trading:")
    print("   - Get Deriv account and API token")
    print("   - Copy .env.example to .env in deriv_trading/")
    print("   - Edit .env with your credentials")
    print("   - Run: python start_trading.py")
    print("   - Choose option 2 (Deriv Trading)")
    print()
    print("3. 🎮 For Demo Trading:")
    print("   - Run: python start_trading.py")
    print("   - Choose option 3 (Demo Trading)")
    print()
    print("4. 📚 Documentation:")
    print("   - Read README.md for complete guide")
    print("   - Check individual README files in subdirectories")

def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install dependencies
    if not install_dependencies():
        return
    
    # Check MT5 installation
    check_mt5_installation()
    
    # Create directories
    create_directories()
    
    # Create example configs
    create_example_configs()
    
    # Check MT5 config
    check_mt5_config()
    
    # Show next steps
    show_next_steps()
    
    print("\n✅ Setup completed successfully!")
    print("🚀 Ready to start trading!")

if __name__ == "__main__":
    main() 