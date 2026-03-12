import cv2
import numpy as np
from datetime import datetime
from human_att.code.florence2Class import Florence2Model
from human_att.code.pipeline import SigLIP

class FrameProcessor:
    def __init__(self):
        self.caption_model = Florence2Model()
        self.classify_model = SigLIP() 
        self.last_frame_time = {}
        self.frame_counters = {}
        
    def process_stream(self, frame_data, camera_id):
        """Process frames for live streaming"""
        frame_rgb = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Calculate FPS
        current_time = datetime.now().timestamp()
        if camera_id not in self.last_frame_time:
            self.last_frame_time[camera_id] = current_time
            fps = 0.0
        else:
            time_diff = current_time - self.last_frame_time[camera_id]
            fps = 2 / time_diff if time_diff > 0 else 0.0
        self.last_frame_time[camera_id] = current_time
        
        # Process frame
        processed_frame, _ = self.classify_model.process_detect(frame_rgb)
        cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        return processed_frame, fps
    
    def process_detailed(self, frame_data):
        """Process frame with detailed analysis"""
        frame_rgb = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        return self.classify_model.process_frame_tasks(frame_rgb)
    
    def process_detailed_with_classification(self, frame_data):
        """Process frame with detailed analysis and classification"""
        frame_rgb = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        processed_frame, caption_results, caption_results, image_results = self.caption_model.process_frame_tasks(frame_rgb)
        _, classes, classes, _ = self.classify_model.process_frame_tasks(frame_rgb)
        return processed_frame, classes, caption_results,  image_results
