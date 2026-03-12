import cv2
import torch
import numpy as np
from ultralytics import YOLO
from queue import Queue
import threading
import argparse
import pickle
from person_attribute_recog_stable_branch.models.model_factory import build_backbone, build_classifier
from datetime import datetime
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

from person_attribute_recog_stable_branch.configs import cfg, update_config
from person_attribute_recog_stable_branch.dataset.pedes_attr.pedes import PedesAttr
from person_attribute_recog_stable_branch.models.base_block import FeatClassifier
# from models.model_factory import model_dict, classifier_dict

from person_attribute_recog_stable_branch.tools.function import get_model_log_path, get_reload_weight
from person_attribute_recog_stable_branch.tools.utils import set_seed, str2bool, time_str
from person_attribute_recog_stable_branch.models.backbone import swin_transformer, resnet, bninception
# from models.backbone.tresnet import tresnet
from torchvision import transforms

class YOLOv8ModelPool:
    def __init__(self, model_names, conf_threshold=0.5):
        self.model_queue = Queue()
        self.conf_threshold = conf_threshold
        self.save_dir = "./save_person"
        # Initialize the attributes_file
        self.attributes_file = os.path.join(self.save_dir, "all_person_attributes.txt")
        # Load multiple models and add them to the queue
        for model_name in model_names:
            self.model_queue.put(YOLO(model_name))
        
        # Initialize attribute recognition model
        self.attribute_model, self.attr_id = self.initialize_attribute_model()

    def initialize_attribute_model(self):
        # Load configuration and dataset info
        cfg = self.load_config()  # You need to implement this method
        dataset_info = pickle.load(open("person_attribute_recog_stable_branch/data/PA100k/dataset_all.pkl", 'rb+'))
        attr_id = dataset_info.attr_name
        # print(attr_id)

        # Build the model
        backbone, c_output = build_backbone(cfg.BACKBONE.TYPE, cfg.BACKBONE.MULTISCALE)
        classifier = build_classifier(cfg.CLASSIFIER.NAME)(
            nattr=len(attr_id),
            c_in=c_output,
            bn=cfg.CLASSIFIER.BN,
            pool=cfg.CLASSIFIER.POOLING,
            scale=cfg.CLASSIFIER.SCALE
        )
        model = FeatClassifier(backbone, classifier)

        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()

        # Load the trained weights
        model = get_reload_weight("exp_result/PA100k/resnet50.base.adam/img_model/ckpt_max_2024-08-24_04_33_15.pth", model)
        model.eval()

        return model, attr_id

    def load_config(self):
        # Implement this method to load your configuration
        # This is a placeholder and needs to be replaced with your actual config loading logic
        class Config:
            class BACKBONE:
                TYPE = "resnet50"
                MULTISCALE = False

            class CLASSIFIER:
                NAME = "linear"
                BN = True
                POOLING = "avg"
                SCALE = 1

        return Config()

    def preprocess_person(self, person_img):
        # tensor = torch.tensor(person_img).permute(2, 0, 1).float()
        transform = transforms.Compose([transforms.ToTensor(),
            transforms.Resize((256, 192)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(person_img).unsqueeze(0)

    def get_attributes(self, person_img):
        results = []
        with torch.no_grad():
            tensor = self.preprocess_person(person_img)
            if torch.cuda.is_available():
                tensor = tensor.cuda()
            logits, _ = self.attribute_model(tensor)
            # print(logits)
            probs = torch.sigmoid(logits[0])
            pred_labels = probs > 0.8
            pred_labels = pred_labels.cpu()
            # print(pred_labels)
            for pred_label in pred_labels:
                result = [attr for attr, pred in zip(self.attr_id, pred_label) if pred]
                # print(result)
                results.append(result)
        return results

    def save_detected_person(self, frame, person_img, attributes):
        # Generate a unique filename for the image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        img_filename = f"person_{timestamp}.jpg"
        img_filepath = os.path.join(self.save_dir, img_filename)

        # Save the original person image without annotations
        cv2.imwrite(img_filepath, person_img)

        # Append attributes to the single text file
        with open(self.attributes_file, 'a') as f:
            f.write(f"Image: {img_filename}\n")
            for attr in attributes:
                attr_str = str(attr)[9:]  # Remove the 'PA100k::' prefix
                f.write(f"- {attr_str}\n")
            f.write("\n")  # Add a blank line between entries

        print(f"Saved detected person to {img_filepath} and appended attributes to {self.attributes_file}")

    def process_frame(self, frame):
        # Acquire a model from the pool
        model = self.model_queue.get()
        attributes = []
        try:
            # Run YOLOv8 model on the frame
            results = model(frame,verbose = False)[0]
            
            # Process the results
            annotated_frame = frame.copy()
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result

                if score > self.conf_threshold:
                    # Extract person image
                    person_img = frame[int(y1):int(y2), int(x1):int(x2)]
                    for result in results.boxes.data.tolist():
                        x1, y1, x2, y2, score, class_id = result

                if score > self.conf_threshold:
                    # Extract person image
                    person_img = frame[int(y1):int(y2), int(x1):int(x2)]
                    # Get attributes
                    attributes = self.get_attributes(person_img)
                    print(attributes)
                    # Save the detected person with attributes
                    # self.save_detected_person(frame, person_img, attributes)
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # Add attributes text
                    y = int(y1) - 10
                    for attr in attributes:
                        attr_str = str(attr)[9:]
                        cv2.putText(annotated_frame, attr_str, (int(x1), y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        y -= 20

        finally:
            # Return the model to the pool
            self.model_queue.put(model)
        
        return annotated_frame, attributes
def main():
    # Initialize the model pool
    model_pool = YOLOv8ModelPool(["yolov8n.pt"])  # You can add more models to the list

    # Open video capture
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide a video file path

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        annotated_frame = model_pool.process_frame(frame)

        # Display the result
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
