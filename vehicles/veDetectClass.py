import torch
import cv2
import time
import os
from ultralytics import YOLO  # YOLOv11 model loader

class VehicleDetector:
    def __init__(self, weight_path="./pretrained/vehicle_detect/yolo11s_veType.pt",
                 vehicle_classes=None, conf_thres=0.5):

        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"
        print(f"Using device for VehicleDetector: {self.device}")
        self.model = YOLO(weight_path)
        self.model.to(self.device)
        self.conf_thres = conf_thres
        self.vehicle_classes = vehicle_classes or ['ô tô', 'xe máy', 'xe tải', 'bus', 'xe đạp']

    def detect(self, img, show=True, save=False, save_dir="runs/detect/vehicle_results"):
        """Run detection on a single image using YOLOv11"""
        results = self.model.predict(
            source=img,
            conf=self.conf_thres,
            save=save,
            project=save_dir if save else None,
            show=show,
            verbose=False,
            device=self.device
        )

        # Extract detections into a pandas-like DataFrame format for compatibility
        detections = []
        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                #name = self.model.names[cls_id] if cls_id in self.model.names else str(cls_id)
                name = self.vehicle_classes[cls_id] if cls_id < len(self.vehicle_classes) else str(cls_id)
                detections.append({
                    "xmin": xyxy[0],
                    "ymin": xyxy[1],
                    "xmax": xyxy[2],
                    "ymax": xyxy[3],
                    "confidence": conf,
                    "class": cls_id,
                    "name": name
                })

        import pandas as pd
        df = pd.DataFrame(detections)
        return df
    
    def drawBoxes(self, rgb_img, listBox_vehicle):
        annotated_frame = rgb_img.copy()
        for box in listBox_vehicle:
            if box[0] == "ô tô":
                color_rgb = (0, 0 ,255)
            else:
                color_rgb = (255, 0, 0)  

            cv2.rectangle(annotated_frame, (box[1], box[2]), (box[3], box[4]), color_rgb, 2)
        return annotated_frame

if __name__ == "__main__":
    detector = VehicleDetector()

    image_path = "405.jpg"
    start_time = time.time()
    detections = detector.detect(image_path, show=False, save=False)
    end_time = time.time()

    # crop each detected object from the original image and show with OpenCV
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
    else:
        for i, det in detections.iterrows():
            xmin = int(det['xmin']); ymin = int(det['ymin'])
            xmax = int(det['xmax']); ymax = int(det['ymax'])
            h, w = img.shape[:2]
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(w, xmax), min(h, ymax)
            if xmax <= xmin or ymax <= ymin:
                continue
            crop = img[ymin:ymax, xmin:xmax].copy()
            cls_name = det['name'] if 'name' in det.index else str(det.get('class', 'obj'))
            conf = det['confidence'] if 'confidence' in det.index else det.get('conf', None)
            if isinstance(conf, (float, int)):
                winname = f"{i}_{cls_name}_{conf:.2f}"
            else:
                winname = f"{i}_{cls_name}"
            cv2.imshow(winname, crop)
            cv2.waitKey(0)  # press any key to advance to the next crop
            cv2.destroyWindow(winname)
        cv2.destroyAllWindows()
    #print(f"Detections:\n{detections}")
    #print(f"Detection Time: {end_time - start_time:.2f} seconds")