#!/usr/bin/env python3
"""
Unified Trading Bot Launcher
============================

This script allows you to choose between MT5 and Deriv trading systems.
"""

import os
import sys
import subprocess
from pathlib import Path

def print_banner():
    """Print the system banner"""
    print("🚀 Unified Trading Bot System")
    print("=" * 50)
    print("📊 Complete Trading Solution for Synthetic Indices")
    print("💰 Optimized for small account scaling ($10+)")
    print("=" * 50)

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔧 Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'python-dotenv', 'flask', 'websocket'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - missing")
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("✅ All dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install some dependencies")
            return False
    
    return True

def check_mt5_installation():
    """Check if MetaTrader5 is available"""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            print("⚠️  MT5 initialization failed. Make sure MetaTrader5 is installed and running.")
            return False
        mt5.shutdown()
        print("✅ MetaTrader5 is available")
        return True
    except ImportError:
        print("❌ MetaTrader5 package not found")
        return False

def show_menu():
    """Show the main menu"""
    print("\n🎯 Choose Your Trading Platform:")
    print("1. 🎮 MT5 Trading (Recommended)")
    print("   - Web interface with mobile access")
    print("   - 20+ trading strategies")
    print("   - Real-time dashboard")
    print("   - Progressive scaling for small accounts")
    print()
    print("2. 🌐 Deriv API Trading")
    print("   - Direct API trading")
    print("   - Multiple strategies")
    print("   - Risk management")
    print("   - Console-based monitoring")
    print()
    print("3. 📊 Demo Trading (MT5)")
    print("   - Safe testing environment")
    print("   - No real money")
    print("   - Full feature testing")
    print()
    print("4. 📚 View Documentation")
    print("   - System overview")
    print("   - Setup instructions")
    print("   - Strategy guide")
    print()
    print("5. 🛑 Exit")
    print()

def start_mt5_trading():
    """Start the MT5 trading system"""
    print("🚀 Starting MT5 Trading System...")
    
    mt5_dir = Path("mt5-trading")
    if not mt5_dir.exists():
        print("❌ MT5 trading directory not found")
        return False
    
    start_script = mt5_dir / "start_unified_web.py"
    if not start_script.exists():
        print("❌ MT5 start script not found")
        return False
    
    try:
        print("📊 Starting web interface...")
        print("🌐 Access: http://localhost:5000")
        print("📱 Mobile: http://[your-ip]:5000")
        print("⚠️  Make sure MT5 is running with AutoTrading enabled!")
        print()
        
        # Change to MT5 directory and start
        os.chdir(mt5_dir)
        subprocess.run([sys.executable, "start_unified_web.py"])
        
    except KeyboardInterrupt:
        print("\n🛑 MT5 trading stopped by user")
    except Exception as e:
        print(f"❌ Error starting MT5 trading: {e}")
        return False
    
    return True

def start_deriv_trading():
    """Start the Deriv trading system"""
    print("🚀 Starting Deriv Trading System...")
    
    deriv_dir = Path("deriv_trading")
    if not deriv_dir.exists():
        print("❌ Deriv trading directory not found")
        return False
    
    main_script = deriv_dir / "main.py"
    if not main_script.exists():
        print("❌ Deriv main script not found")
        return False
    
    # Check for .env file
    env_file = deriv_dir / ".env"
    if not env_file.exists():
        print("⚠️  No .env file found in deriv_trading/")
        print("📝 Please create .env file with your Deriv credentials:")
        print("   DERIV_APP_ID=your_app_id")
        print("   DERIV_ACCOUNT_TOKEN=your_token")
        print("   DERIV_IS_DEMO=true")
        print("   DERIV_DEFAULT_AMOUNT=1.00")
        print()
        create_env = input("Create .env file now? (y/n): ").lower()
        if create_env == 'y':
            create_deriv_env()
    
    try:
        print("📊 Starting Deriv trading bot...")
        print("📈 Real-time console output")
        print("🛡️  Risk management active")
        print()
        
        # Change to Deriv directory and start
        os.chdir(deriv_dir)
        subprocess.run([sys.executable, "main.py"])
        
    except KeyboardInterrupt:
        print("\n🛑 Deriv trading stopped by user")
    except Exception as e:
        print(f"❌ Error starting Deriv trading: {e}")
        return False
    
    return True

def start_demo_trading():
    """Start the demo trading system"""
    print("🚀 Starting Demo Trading System...")
    
    demo_dir = Path("mt5-trading/demo-trading")
    if not demo_dir.exists():
        print("❌ Demo trading directory not found")
        return False
    
    run_script = demo_dir / "run-demo.py"
    if not run_script.exists():
        print("❌ Demo run script not found")
        return False
    
    try:
        print("🎮 Starting demo trading...")
        print("🟢 Safe testing environment")
        print("💰 No real money involved")
        print()
        
        # Change to demo directory and start
        os.chdir(demo_dir)
        subprocess.run([sys.executable, "run-demo.py"])
        
    except KeyboardInterrupt:
        print("\n🛑 Demo trading stopped by user")
    except Exception as e:
        print(f"❌ Error starting demo trading: {e}")
        return False
    
    return True

def create_deriv_env():
    """Create a basic .env file for Deriv trading"""
    env_content = """# Deriv Trading Configuration
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
    env_file = deriv_dir / ".env"
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print(f"✅ Created {env_file}")
        print("📝 Please edit the file with your actual credentials")
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")

def show_documentation():
    """Show documentation options"""
    print("\n📚 Documentation Options:")
    print("1. 📖 System Overview (README.md)")
    print("2. 🎮 MT5 Trading Guide")
    print("3. 🌐 Deriv Trading Guide")
    print("4. 📊 Strategy Guide")
    print("5. 🔧 Setup Guide")
    print("6. ⬅️  Back to Main Menu")
    print()
    
    choice = input("Select option (1-6): ").strip()
    
    if choice == "1":
        print("\n📖 System Overview:")
        print("This is a comprehensive trading bot system with two main components:")
        print("- MT5 Trading: MetaTrader 5-based system with web interface")
        print("- Deriv Trading: Deriv API-based system with console interface")
        print("Both systems are optimized for synthetic indices trading.")
        print("\nKey features:")
        print("- 20+ trading strategies")
        print("- Advanced risk management")
        print("- Progressive scaling for small accounts")
        print("- Web dashboard with mobile access")
        print("- Demo and real trading modes")
    
    elif choice == "2":
        print("\n🎮 MT5 Trading Guide:")
        print("1. Install MetaTrader 5 terminal")
        print("2. Enable AutoTrading in MT5")
        print("3. Run: cd mt5-trading && python start_unified_web.py")
        print("4. Access dashboard at http://localhost:5000")
        print("5. Connect MT5 and start trading")
    
    elif choice == "3":
        print("\n🌐 Deriv Trading Guide:")
        print("1. Get Deriv account and API token")
        print("2. Create .env file with credentials")
        print("3. Run: cd deriv_trading && python main.py")
        print("4. Monitor console output")
    
    elif choice == "4":
        print("\n📊 Strategy Guide:")
        print("Available strategies:")
        print("- Moving Average Crossover")
        print("- RSI Strategy")
        print("- Bollinger Bands")
        print("- MACD Strategy")
        print("- And 15+ more...")
        print("See mt5-trading/strategies_mt5.py for details")
    
    elif choice == "5":
        print("\n🔧 Setup Guide:")
        print("1. Install Python 3.7+")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Configure account credentials")
        print("4. Start with demo trading")
        print("5. Test thoroughly before real trading")
    
    input("\nPress Enter to continue...")

def main():
    """Main function"""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Failed to install dependencies")
        return
    
    # Check MT5 availability
    mt5_available = check_mt5_installation()
    
    while True:
        show_menu()
        
        choice = input("Select option (1-5): ").strip()
        
        if choice == "1":
            if not mt5_available:
                print("❌ MetaTrader5 not available")
                print("💡 Install MetaTrader5 package: pip install MetaTrader5")
                continue
            start_mt5_trading()
            break
            
        elif choice == "2":
            start_deriv_trading()
            break
            
        elif choice == "3":
            if not mt5_available:
                print("❌ MetaTrader5 not available")
                print("💡 Install MetaTrader5 package: pip install MetaTrader5")
                continue
            start_demo_trading()
            break
            
        elif choice == "4":
            show_documentation()
            
        elif choice == "5":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid option. Please try again.")

if __name__ == "__main__":
    main() 