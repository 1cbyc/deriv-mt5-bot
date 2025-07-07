import MetaTrader5 as mt5
import os
from dotenv import load_dotenv

def test_mt5_connection():
    """Test MT5 connection and list available symbols"""
    
    # Load environment variables
    load_dotenv()
    
    # Initialize MT5
    if not mt5.initialize():
        print(f"❌ MT5 initialization failed: {mt5.last_error()}")
        return False
    
    # Login
    login = int(os.getenv('MT5_LOGIN', '0'))
    password = os.getenv('MT5_PASSWORD', '')
    server = os.getenv('MT5_SERVER', 'Deriv-Demo')
    
    if not mt5.login(login=login, password=password, server=server):
        print(f"❌ MT5 login failed: {mt5.last_error()}")
        return False
    
    print(f"✅ Connected to MT5 account: {login}")
    
    # Get account info
    account_info = mt5.account_info()
    if account_info:
        print(f"💰 Balance: ${account_info.balance:.2f}")
        print(f"🏢 Broker: {account_info.company}")
    
    # List all symbols
    print("\n📊 Available Symbols:")
    print("=" * 50)
    
    symbols = mt5.symbols_get()
    if symbols:
        all_symbol_names = [symbol.name for symbol in symbols]
        for symbol_name in all_symbol_names:
            print(symbol_name)
        print(f"\n📈 Total Symbols Available: {len(all_symbol_names)}")
    else:
        print("❌ No symbols found")
    
    # Cleanup
    mt5.shutdown()
    return True

if __name__ == "__main__":
    test_mt5_connection() 