import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import cv2
from PIL import Image
import os
from ultralytics import YOLO
from typing import List, Tuple, Dict
import numpy as np


def xywh_to_x1y1x2y2(x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return x1, y1, x2, y2


class BatchFlorence2Model:
    def __init__(self, model_name: str = "microsoft/Florence-2-base", batch_size: int = 4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.detection_model = YOLO("/home/t4-vkist/ETRI-SUB/video_management_system/pretrained/yolov8n.pt")
        self.torch_dtype = torch.float32
        self.conf_threshold = 0.3
        self.batch_size = batch_size
        
        # Initialize Florence-2 model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True
        ).to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

    def generate_batch(self, task_prompt: str, images: List[Image.Image], text_inputs: List[str] = None) -> List[Dict]:
        """Process a batch of images with Florence-2 model"""
        if text_inputs is None:
            text_inputs = [task_prompt] * len(images)
        else:
            text_inputs = [task_prompt + text_input for text_input in text_inputs]

        # Process in batches
        all_results = []
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            batch_texts = text_inputs[i:i + self.batch_size]
            
            inputs = self.processor(
                text=batch_texts,
                images=batch_images,
                return_tensors="pt",
                padding=True
            )

            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"].to(self.device),
                pixel_values=inputs["pixel_values"].to(self.device),
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=3,
            )

            generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=False)
            
            # Post-process each generated text
            batch_results = []
            for j, generated_text in enumerate(generated_texts):
                parsed_answer = self.processor.post_process_generation(
                    generated_text,
                    task=task_prompt,
                    image_size=(batch_images[j].width, batch_images[j].height)
                )
                batch_results.append(parsed_answer)
                
            all_results.extend(batch_results)
            
        return all_results

    def process_frames_batch(self, 
                           frames: List[np.ndarray], 
                           path_img_out: str = "output_directory/") -> Tuple[List[np.ndarray], List[Dict], List[List[Dict]]]:
        """Process a batch of frames with both YOLO and Florence-2"""
        
        # Convert frames to PIL Images for Florence-2
        pil_images = [Image.fromarray(frame) for frame in frames]
        
        # Batch detection with YOLO
        detection_results = self.detection_model(frames, verbose=False)
        
        # Process each frame's detections and prepare crops
        annotated_frames = []
        all_crops = []
        crop_coordinates = []  # Store coordinates for mapping results back
        frame_indices = []     # Store frame indices for mapping results back
        
        for frame_idx, (frame, detection) in enumerate(zip(frames, detection_results)):
            annotated_frame = frame.copy()
            frame_crops = []
            frame_coords = []
            
            # Process person detections
            for box in detection.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = box
                if class_id == 0 and score > self.conf_threshold:
                    # Draw rectangle on annotated frame
                    cv2.rectangle(annotated_frame, 
                                (int(x1), int(y1)), 
                                (int(x2), int(y2)), 
                                (0, 255, 0), 2)
                    
                    # Crop person
                    crop = pil_images[frame_idx].crop([x1, y1, x2, y2])
                    frame_crops.append(crop)
                    frame_coords.append([x1, y1, x2, y2])
                    frame_indices.append(frame_idx)
            
            annotated_frames.append(annotated_frame)
            all_crops.extend(frame_crops)
            crop_coordinates.extend(frame_coords)
        
        # Generate captions for full frames
        frame_captions = self.generate_batch('<CAPTION>', pil_images)
        
        # Generate detailed captions for all crops
        if all_crops:  # Only process if we have detected persons
            crop_captions = self.generate_batch('<DETAILED_CAPTION>', all_crops)
            
            # Reorganize crop captions by frame
            frame_crop_captions = [[] for _ in frames]
            for caption_idx, frame_idx in enumerate(frame_indices):
                frame_crop_captions[frame_idx].append({
                    'caption': crop_captions[caption_idx],
                    'bbox': crop_coordinates[caption_idx]
                })
        else:
            frame_crop_captions = [[] for _ in frames]
        
        return annotated_frames, frame_captions, frame_crop_captions

    def save_results(self, 
                    frame_idx: int,
                    annotated_frame: np.ndarray,
                    frame_caption: Dict,
                    crop_captions: List[Dict],
                    path_img_out: str):
        """Save processing results to files"""
        os.makedirs(path_img_out, exist_ok=True)
        
        # Save annotated frame
        frame_path = os.path.join(path_img_out, f"frame_{frame_idx}.jpg")
        cv2.imwrite(frame_path, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
        
        # Save frame caption
        frame_caption_path = os.path.join(path_img_out, f"frame_{frame_idx}_caption.txt")
        with open(frame_caption_path, 'w') as f:
            f.write(str(frame_caption))
        
        # Save crop captions
        for i, crop_data in enumerate(crop_captions):
            crop_caption_path = os.path.join(path_img_out, f"frame_{frame_idx}_crop_{i}.txt")
            with open(crop_caption_path, 'w') as f:
                f.write(str(crop_data))


# Example usage
def process_video_batch(video_path: str, output_dir: str, batch_size: int = 4):
    model = BatchFlorence2Model(batch_size=batch_size)
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        
        # Process when we have a full batch
        if len(frames) == batch_size:
            annotated_frames, frame_captions, crop_captions = model.process_frames_batch(frames)
            
            # Save results for each frame in the batch
            for i in range(batch_size):
                model.save_results(
                    frame_count + i,
                    annotated_frames[i],
                    frame_captions[i],
                    crop_captions[i],
                    output_dir
                )
            
            frames = []  # Clear the batch
            frame_count += batch_size
        
    # Process any remaining frames
    if frames:
        annotated_frames, frame_captions, crop_captions = model.process_frames_batch(frames)
        for i in range(len(frames)):
            model.save_results(
                frame_count + i,
                annotated_frames[i],
                frame_captions[i],
                crop_captions[i],
                output_dir
            )
    
    cap.release()