import json
import time
import websocket
import threading
from typing import Dict, List, Callable, Optional
from .config import DerivConfig

class DerivAPIClient:
    """WebSocket client for Deriv API"""
    
    def __init__(self):
        self.ws = None
        self.connected = False
        self.callbacks = {}
        self.pending_requests = {}
        self.request_id = 0
        self.account_balance = 0.0
        self.account_currency = 'USD'
        
    def connect(self):
        """Establish WebSocket connection to Deriv API"""
        try:
            self.ws = websocket.WebSocketApp(
                DerivConfig.API_URL,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            # Start WebSocket connection in a separate thread
            self.ws_thread = threading.Thread(target=self.ws.run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            # Wait for connection
            timeout = 10
            while not self.connected and timeout > 0:
                time.sleep(0.1)
                timeout -= 0.1
                
            if not self.connected:
                raise Exception("Failed to connect to Deriv API")
                
            print("✅ Connected to Deriv API")
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def _on_open(self, ws):
        """Handle WebSocket connection open"""
        print("🔗 WebSocket connection opened")
        self.connected = True
        
        # Authorize the connection
        self.authorize()
    
    def _on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            
            # Handle authorization response
            if 'msg_type' in data and data['msg_type'] == 'authorize':
                if data.get('error'):
                    print(f"❌ Authorization failed: {data['error']['message']}")
                else:
                    print("✅ Authorization successful")
                    account_info = data.get('authorize', {})
                    balance = float(account_info.get('balance', 0))
                    currency = account_info.get('currency', 'USD')
                    print(f"💰 Balance: {balance} {currency}")
                    print(f"🏦 Account: {account_info.get('loginid', 'N/A')}")
                    
                    # Store balance for later use
                    self.account_balance = balance
                    self.account_currency = currency
            
            # Handle other responses
            elif 'req_id' in data:
                req_id = data['req_id']
                if req_id in self.pending_requests:
                    callback = self.pending_requests.pop(req_id)
                    if callback:
                        callback(data)
            
            # Handle error messages
            elif 'error' in data:
                print(f"❌ API Error: {data['error']['message']}")
                
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse message: {e}")
    
    def _on_error(self, ws, error):
        """Handle WebSocket errors"""
        print(f"❌ WebSocket error: {error}")
        self.connected = False
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close"""
        print("🔌 WebSocket connection closed")
        self.connected = False
    
    def authorize(self):
        """Authorize the API connection"""
        if DerivConfig.ACCOUNT_TOKEN:
            self.send_request({
                "authorize": DerivConfig.ACCOUNT_TOKEN
            })
        else:
            print("⚠️  No account token provided - using public connection")
    
    def send_request(self, request: Dict, callback: Optional[Callable] = None):
        """Send a request to the Deriv API"""
        if not self.connected:
            print("❌ Not connected to API")
            return None
        
        # Add request ID
        self.request_id += 1
        request['req_id'] = self.request_id
        
        # Store callback if provided
        if callback:
            self.pending_requests[self.request_id] = callback
        
        # Send request
        try:
            self.ws.send(json.dumps(request))  # type: ignore
            return self.request_id
        except Exception as e:
            print(f"❌ Failed to send request: {e}")
            return None
    
    def get_ticks(self, symbol: str, callback: Callable):
        """Subscribe to price ticks for a symbol"""
        request = {
            "ticks": symbol,
            "subscribe": 1
        }
        return self.send_request(request, callback)
    
    def get_candles(self, symbol: str, granularity: int, count: int = 1000, callback: Optional[Callable] = None):
        """Get historical candles for a symbol"""
        request = {
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": count,
            "end": "latest",
            "start": int(time.time()) - (granularity * count),
            "style": "candles",
            "granularity": granularity
        }
        return self.send_request(request, callback)  # type: ignore
    
    def buy_contract(self, symbol: str, amount: float, direction: str, duration: int, callback: Optional[Callable] = None):
        """Buy a binary options contract (Rise/Fall) for Deriv synthetic indices"""
        # Deriv expects 'price' as the stake amount at the top level, not in parameters
        request = {
            "buy": 1,
            "price": amount,  # Stake amount (USD)
            "parameters": {
                "amount": amount,
                "basis": "stake",
                "contract_type": "CALL" if direction.upper() == "UP" else "PUT",
                "currency": DerivConfig.DEFAULT_CURRENCY,
                "duration": duration,
                "duration_unit": "t",
                "symbol": symbol
            }
        }
        return self.send_request(request, callback)  # type: ignore
    
    def get_balance(self, callback: Optional[Callable] = None):
        """Get account balance"""
        request = {
            "get_settings": 1
        }
        return self.send_request(request, callback)  # type: ignore
    
    def disconnect(self):
        """Disconnect from the API"""
        if self.ws:
            self.ws.close()
        self.connected = False
        print("🔌 Disconnected from Deriv API") 