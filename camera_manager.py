import threading
from datetime import datetime

class CameraManager:
    def __init__(self):
        self.latest_frames = {}
        self.frame_locks = {}
        self.message_counter = {}
        
    def update_frame(self, camera_id, frame_data, fps):
        if camera_id not in self.frame_locks:
            self.frame_locks[camera_id] = threading.Lock()
            
        with self.frame_locks[camera_id]:
            self.latest_frames[camera_id] = {
                'frame_data': frame_data,
                'timestamp': datetime.now().isoformat(),
                'fps': fps
            }
    
    def get_frame(self, camera_id):
        if camera_id not in self.latest_frames:
            return None
            
        with self.frame_locks[camera_id]:
            return self.latest_frames[camera_id].copy()