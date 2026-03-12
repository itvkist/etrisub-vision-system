from surveillance_system import SurveillanceSystem
from redis_manager import RedisManager

if __name__ == '__main__':
    # Start Redis server
    redis_manager = RedisManager()
    if redis_manager.start():
        try:
            # Start surveillance system
            surveillance_system = SurveillanceSystem()
            surveillance_system.run()
        finally:
            # Ensure Redis is stopped on exit
            redis_manager.stop()
