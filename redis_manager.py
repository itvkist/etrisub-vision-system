import subprocess
import os
import signal
import time

class RedisManager:
    def __init__(self, config_path='config/redis.conf'):
        self.config_path = config_path
        self.redis_process = None
        
    def start(self):
        try:
            # Check if Redis is already running
            result = subprocess.run(['pgrep', 'redis-server'], capture_output=True)
            if result.returncode == 0:
                print("Redis server is already running")
                return True
                
            # Start Redis with our config
            self.redis_process = subprocess.Popen(
                ['redis-server', self.config_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait a moment to ensure Redis starts
            time.sleep(2)
            
            if self.redis_process.poll() is None:
                print("Redis server started successfully")
                return True
            else:
                print("Failed to start Redis server")
                return False
                
        except Exception as e:
            print(f"Error starting Redis: {str(e)}")
            return False
            
    def stop(self):
        try:
            if self.redis_process:
                self.redis_process.terminate()
                self.redis_process.wait(timeout=5)
            
            # Force kill any remaining Redis process
            subprocess.run(['pkill', 'redis-server'])
            print("Redis server stopped")
            
        except Exception as e:
            print(f"Error stopping Redis: {str(e)}")
            
    def __del__(self):
        self.stop()
