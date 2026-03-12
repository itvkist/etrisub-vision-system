import torch
from ultralytics import YOLO

class HumanDetector:
    def __init__(self, weight_path="../../pretrained/yolov8n.pt",
                 conf_thres=0.5):

        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"
        print(f"Using device for HumanDetector: {self.device}")
        self.model = YOLO(weight_path)
        self.model.to(self.device)
        self.conf_thres = conf_thres

    def detect(self, img, show=False, save=False, save_dir="runs/detect/human_results"):
        """Run detection on a single image using YOLOv8"""
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
                name = self.model.names[cls_id] if cls_id in self.model.names else str(cls_id)
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
    
if __name__ == "__main__":
    detector = HumanDetector()
    #import cv2
    #img = cv2.imread("test_images/human_test.jpg")
    #results = detector.detect(img, show=True)
    #print(results)
    print("HumanDetector is ready.")