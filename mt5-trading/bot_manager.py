#!/usr/bin/env python3
"""
MT5 Bot Manager
===============

Manages MT5 trading bots as separate processes.
This avoids import issues and provides better isolation.
"""

import subprocess
import os
import sys
import time
import signal
import psutil
from datetime import datetime
from typing import Dict, Optional, List
import json

class BotManager:
    """Manages MT5 trading bot processes"""
    
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.status: Dict[str, str] = {}
        self.logs: List[str] = []
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
    def _log(self, message: str):
        """Add a log message with timestamp"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        print(log_entry)
        
    def start_bot(self, bot_type: str) -> tuple[bool, str]:
        """Start a bot of the specified type"""
        if bot_type in self.processes and self.processes[bot_type].poll() is None:
            return False, f"{bot_type} bot is already running"
        
        try:
            # Determine the script path based on bot type
            if bot_type == "demo":
                script_path = os.path.join(self.base_dir, "demo-trading", "main.py")
            elif bot_type == "real":
                script_path = os.path.join(self.base_dir, "real-trading", "main-real.py")
            elif bot_type == "micro":
                script_path = os.path.join(self.base_dir, "real-trading", "main-micro-scaling.py")
            else:
                return False, f"Unknown bot type: {bot_type}"
            
            # Check if script exists
            if not os.path.exists(script_path):
                return False, f"Bot script not found: {script_path}"
            
            # Start the bot process
            self._log(f"Starting {bot_type} bot...")
            
            process = subprocess.Popen(
                [sys.executable, script_path],
                cwd=self.base_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self.processes[bot_type] = process
            self.status[bot_type] = "starting"
            
            # Wait a moment to see if it starts successfully
            time.sleep(2)
            
            if process.poll() is None:
                self.status[bot_type] = "running"
                self._log(f"{bot_type} bot started successfully (PID: {process.pid})")
                return True, f"{bot_type} bot started successfully"
            else:
                # Process died immediately
                stdout, stderr = process.communicate()
                error_msg = f"{bot_type} bot failed to start"
                if stderr:
                    error_msg += f": {stderr.strip()}"
                self._log(error_msg)
                return False, error_msg
                
        except Exception as e:
            error_msg = f"Failed to start {bot_type} bot: {str(e)}"
            self._log(error_msg)
            return False, error_msg
    
    def stop_bot(self, bot_type: str) -> tuple[bool, str]:
        """Stop a specific bot"""
        if bot_type not in self.processes:
            return False, f"No {bot_type} bot running"
        
        process = self.processes[bot_type]
        
        try:
            self._log(f"Stopping {bot_type} bot...")
            
            # Try graceful termination first
            process.terminate()
            
            # Wait for graceful shutdown
            try:
                process.wait(timeout=10)
                self._log(f"{bot_type} bot stopped gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown fails
                self._log(f"Force killing {bot_type} bot...")
                process.kill()
                process.wait()
                self._log(f"{bot_type} bot force killed")
            
            del self.processes[bot_type]
            self.status[bot_type] = "stopped"
            return True, f"{bot_type} bot stopped successfully"
            
        except Exception as e:
            error_msg = f"Error stopping {bot_type} bot: {str(e)}"
            self._log(error_msg)
            return False, error_msg
    
    def stop_all_bots(self) -> tuple[bool, str]:
        """Stop all running bots"""
        if not self.processes:
            return True, "No bots running"
        
        results = []
        for bot_type in list(self.processes.keys()):
            success, message = self.stop_bot(bot_type)
            results.append(f"{bot_type}: {message}")
        
        return True, "; ".join(results)
    
    def get_status(self) -> Dict:
        """Get status of all bots"""
        # Update status based on process state
        for bot_type, process in self.processes.items():
            if process.poll() is None:
                self.status[bot_type] = "running"
            else:
                self.status[bot_type] = "stopped"
                # Clean up dead process
                if bot_type in self.processes:
                    del self.processes[bot_type]
        
        return {
            'bots': self.status,
            'logs': self.logs[-20:],  # Last 20 logs
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def get_bot_info(self, bot_type: str) -> Dict:
        """Get detailed info about a specific bot"""
        if bot_type not in self.processes:
            return {'status': 'not_running', 'pid': None}
        
        process = self.processes[bot_type]
        if process.poll() is None:
            try:
                # Get process info
                proc = psutil.Process(process.pid)
                return {
                    'status': 'running',
                    'pid': process.pid,
                    'cpu_percent': proc.cpu_percent(),
                    'memory_mb': proc.memory_info().rss / 1024 / 1024,
                    'create_time': datetime.fromtimestamp(proc.create_time()).strftime('%H:%M:%S')
                }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return {'status': 'unknown', 'pid': process.pid}
        else:
            return {'status': 'stopped', 'pid': None}

# Global bot manager instance
bot_manager = BotManager() 