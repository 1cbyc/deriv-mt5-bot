#!/usr/bin/env python3
"""
Simple MT5 Trading Bot Web Interface
====================================

A simplified Flask web application to test basic functionality.
"""

from flask import Flask, render_template, jsonify, request
import os
import sys
from datetime import datetime
import subprocess
import threading
import time

app = Flask(__name__)

# Global state
bot_process = None
bot_status = "stopped"
bot_logs = []

class SimpleBotManager:
    """Simple bot manager that runs bots as separate processes"""
    
    def __init__(self):
        self.process = None
        self.status = "stopped"
        self.logs = []
        
    def start_demo_bot(self):
        """Start demo bot as a separate process"""
        if self.status == "running":
            return False, "Bot is already running"
        
        try:
            # Start demo bot as a separate process
            self.process = subprocess.Popen(
                [sys.executable, "demo-trading/main.py"],
                cwd=os.path.dirname(__file__),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.status = "running"
            self.logs.append(f"Demo bot started at {datetime.now()}")
            return True, "Demo bot started successfully"
            
        except Exception as e:
            error_msg = f"Failed to start demo bot: {str(e)}"
            self.logs.append(error_msg)
            return False, error_msg
    
    def stop_bot(self):
        """Stop the bot process"""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
                self.status = "stopped"
                self.logs.append(f"Bot stopped at {datetime.now()}")
                return True, "Bot stopped successfully"
            except Exception as e:
                error_msg = f"Error stopping bot: {str(e)}"
                self.logs.append(error_msg)
                return False, error_msg
        return False, "No bot running"
    
    def get_status(self):
        """Get current bot status"""
        if self.process:
            # Check if process is still running
            if self.process.poll() is None:
                self.status = "running"
            else:
                self.status = "stopped"
        
        return {
            'status': self.status,
            'logs': self.logs[-10:],  # Last 10 logs
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

# Create bot manager
bot_manager = SimpleBotManager()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('simple_dashboard.html')

@app.route('/api/status')
def get_status():
    """API endpoint to get bot status"""
    return jsonify(bot_manager.get_status())

@app.route('/api/start', methods=['POST'])
def start_bot():
    """API endpoint to start the bot"""
    data = request.get_json() or {}
    bot_type = data.get('bot_type', 'demo')
    
    if bot_type == 'demo':
        success, message = bot_manager.start_demo_bot()
    else:
        success, message = False, f"Unknown bot type: {bot_type}"
    
    return jsonify({'success': success, 'message': message})

@app.route('/api/stop', methods=['POST'])
def stop_bot():
    """API endpoint to stop the bot"""
    success, message = bot_manager.stop_bot()
    return jsonify({'success': success, 'message': message})

@app.route('/api/config')
def get_config():
    """Get available configuration options"""
    return jsonify({
        'bot_types': [
            {'id': 'demo', 'name': 'Demo Trading Bot', 'description': 'Safe demo account trading'}
        ]
    })

def create_simple_template():
    """Create a simple HTML template"""
    os.makedirs('templates', exist_ok=True)
    
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simple MT5 Trading Bot</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .status {
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            font-weight: bold;
        }
        .status.running { background: #d4edda; color: #155724; }
        .status.stopped { background: #f8d7da; color: #721c24; }
        .btn {
            padding: 10px 20px;
            margin: 5px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        .btn-success { background: #28a745; color: white; }
        .btn-danger { background: #dc3545; color: white; }
        .logs {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            max-height: 300px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Simple MT5 Trading Bot</h1>
        
        <div class="status" id="statusDisplay">
            Status: <span id="botStatus">Loading...</span>
        </div>
        
        <div>
            <button class="btn btn-success" onclick="startBot()">Start Demo Bot</button>
            <button class="btn btn-danger" onclick="stopBot()">Stop Bot</button>
        </div>
        
        <div id="message"></div>
        
        <h3>Bot Logs</h3>
        <div class="logs" id="logsDisplay"></div>
    </div>
    
    <script>
        function updateStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('botStatus').textContent = data.status;
                    document.getElementById('statusDisplay').className = 'status ' + data.status;
                    
                    const logsDiv = document.getElementById('logsDisplay');
                    logsDiv.innerHTML = data.logs.join('<br>');
                    logsDiv.scrollTop = logsDiv.scrollHeight;
                })
                .catch(error => console.error('Error:', error));
        }
        
        function startBot() {
            fetch('/api/start', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({bot_type: 'demo'})
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('message').innerHTML = 
                    '<div style="padding: 10px; margin: 10px 0; background: ' + 
                    (data.success ? '#d4edda' : '#f8d7da') + '; color: ' + 
                    (data.success ? '#155724' : '#721c24') + '; border-radius: 5px;">' + 
                    data.message + '</div>';
            })
            .catch(error => console.error('Error:', error));
        }
        
        function stopBot() {
            fetch('/api/stop', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('message').innerHTML = 
                    '<div style="padding: 10px; margin: 10px 0; background: ' + 
                    (data.success ? '#d4edda' : '#f8d7da') + '; color: ' + 
                    (data.success ? '#155724' : '#721c24') + '; border-radius: 5px;">' + 
                    data.message + '</div>';
            })
            .catch(error => console.error('Error:', error));
        }
        
        // Update status every 2 seconds
        setInterval(updateStatus, 2000);
        updateStatus();
    </script>
</body>
</html>'''
    
    with open('templates/simple_dashboard.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == '__main__':
    create_simple_template()
    
    print("🌐 Starting Simple MT5 Trading Bot Web Interface...")
    print("📱 Access from your phone at: http://[YOUR_PC_IP]:5000")
    print("💻 Access from your PC at: http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=False) 