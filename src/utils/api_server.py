"""Utility to manage FastAPI server for Expense Manager APIs."""

import threading
import uvicorn
import atexit
import requests
import time
from typing import Optional


class APIServerManager:
    """Manages the FastAPI server process."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8000):
        self.host = host
        self.port = port
        self.server_thread: Optional[threading.Thread] = None
        self.server_running = False
        self.config = None
        
    def _run_server(self):
        """Run the uvicorn server in the thread."""
        try:
            config = uvicorn.Config(
                "src.api.main:app",
                host=self.host,
                port=self.port,
                log_level="warning",  # Reduce log noise
                access_log=False
            )
            server = uvicorn.Server(config)
            self.config = config
            self.server_running = True
            server.run()
        except Exception as e:
            print(f"[API Server] Error: {e}")
            self.server_running = False
    
    def start(self):
        """Start the API server in a background thread."""
        if self.is_running():
            print(f"[API Server] Already running at http://{self.host}:{self.port}")
            return True
        
        try:
            self.server_thread = threading.Thread(
                target=self._run_server,
                daemon=True,  # Daemon thread will die when main thread dies
                name="FastAPI-Server"
            )
            self.server_thread.start()
            
            # Wait a bit and check if server started
            max_attempts = 10
            for i in range(max_attempts):
                time.sleep(0.5)
                if self.is_running():
                    print(f"[API Server] Started successfully at http://{self.host}:{self.port}")
                    return True
                if i == max_attempts - 1:
                    print(f"[API Server] Warning: Server may not have started correctly")
                    return False
            return True
        except Exception as e:
            print(f"[API Server] Failed to start: {e}")
            return False
    
    def stop(self):
        """Stop the API server."""
        # Since it's a daemon thread, it will stop automatically when main process ends
        # But we can mark it as stopped
        self.server_running = False
        print("[API Server] Stopped")
    
    def is_running(self) -> bool:
        """Check if the API server is running by making a health check request."""
        try:
            response = requests.get(
                f"http://{self.host}:{self.port}/api/health",
                timeout=1
            )
            return response.status_code == 200
        except:
            return False
    
    def get_url(self) -> str:
        """Get the base URL of the API server."""
        return f"http://{self.host}:{self.port}"


# Global server manager instance
_server_manager: Optional[APIServerManager] = None


def get_api_server_manager(host: str = "127.0.0.1", port: int = 8000) -> APIServerManager:
    """Get or create the global API server manager."""
    global _server_manager
    if _server_manager is None:
        _server_manager = APIServerManager(host=host, port=port)
        # Register cleanup on exit
        atexit.register(_server_manager.stop)
    return _server_manager


def start_api_server(host: str = "127.0.0.1", port: int = 8000) -> bool:
    """Start the API server if not already running."""
    manager = get_api_server_manager(host=host, port=port)
    if not manager.is_running():
        return manager.start()
    return True


def is_api_server_running() -> bool:
    """Check if the API server is running."""
    if _server_manager is None:
        return False
    return _server_manager.is_running()


def get_api_server_url() -> str:
    """Get the API server URL."""
    if _server_manager is None:
        return "http://127.0.0.1:8000"
    return _server_manager.get_url()
