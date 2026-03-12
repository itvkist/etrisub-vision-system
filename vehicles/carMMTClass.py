import torch
import torch.nn as nn
import torchvision
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

VN_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

from vehicles.veDetectClass import VehicleDetector
from torchvision import transforms
from vehicles.utils import load_class_names, separate_class

class NetworkV2(nn.Module):
    def __init__(self, base, num_classes, num_makes, num_types):
        super().__init__()
        self.base = base

        if hasattr(base, 'fc'):
            in_features = self.base.fc.in_features
            self.base.fc = nn.Sequential()
        else:  # mobile net v2
            in_features = self.base.last_channel
            self.base.classifier = nn.Sequential()

        self.brand_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_makes)
        )

        self.type_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_types)
        )

        self.class_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(in_features + num_makes + num_types, num_classes)
        )

    def forward(self, x):
        out = self.base(x)
        brand_fc = self.brand_fc(out)
        type_fc = self.type_fc(out)

        concat = torch.cat([out, brand_fc, type_fc], dim=1)

        fc = self.class_fc(concat)

        return fc, brand_fc, type_fc

class NetworkV3(nn.Module):
    def __init__(self, base, num_classes, num_makes, num_types):
        super().__init__()
        self.base = base

        if hasattr(base, 'fc'):
            in_features = self.base.fc.in_features
            self.base.fc = nn.Sequential()
        else:  # mobile net v2
            in_features = self.base.last_channel
            self.base.classifier = nn.Sequential()

        self.brand_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(num_classes, num_makes)
        )

        self.type_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(num_classes, num_types)
        )

        self.class_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        out = self.base(x)
        fc = self.class_fc(out)
        brand_fc = self.brand_fc(fc)
        type_fc = self.type_fc(fc)

        return fc, brand_fc, type_fc

class CarMMTRecognizer:
    def __init__(self, arch='resnext50', version=2, weight_path='./pretrained/car_att/carMMT_pnk.pth', font_path=VN_FONT_PATH):
        
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.detector = VehicleDetector()
        print(f"Using device for CarMMTRecognizer: {self.device}")
        self.class_names = load_class_names()
        num_classes = len(self.class_names)
        v2_info = separate_class(self.class_names)
        self.make_names = v2_info['make'].unique()
        num_makes = len(self.make_names)
        self.model_type_names = v2_info['model_type'].unique()
        num_types = len(self.model_type_names)

        #self.font = cv2.freeType.createFreeType2()
        #self.font.loadFontData(fontFileName="./fonts/VNTime.ttf", id=0)
        self.font_path = font_path
        self.font_height = 20

        if arch == 'resnext50':
            base = torchvision.models.resnext50_32x4d(pretrained=True)
        else:
            base = torchvision.models.mobilenet_v2(pretrained=True)
        if version == 2:
            self.recognizer = NetworkV2(base, num_classes, num_makes, num_types)
        else:
            self.recognizer = NetworkV3(base, num_classes, num_makes, num_types)

        sd = torch.load(weight_path, map_location=self.device)
        self.recognizer.load_state_dict(sd)
        self.recognizer = self.recognizer.to(self.device)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4706145, 0.46000465, 0.45479808),
                (0.26668432, 0.26578658, 0.2706199)
            )
        ])

    def process_detect(self, rgb_frame):
        detection_results = self.detector.detect(rgb_frame, show=False)
        
        boxes = []
        for _, row in detection_results.iterrows():
            boxes.append([
                row['name'],
                int(row['xmin']),
                int(row['ymin']),
                int(row['xmax']),
                int(row['ymax'])
            ])
        
        return boxes
    
    def process_frame_tasks(self, rgb_frame):
        boxes = self.process_detect(rgb_frame)
        pil_image = Image.fromarray(rgb_frame)
        attributes = []
        crops = []

        for box in boxes:
            crop = pil_image.crop(box[1:])
            crops.append(np.array(crop))

            if box[0] != "ô tô":
                attributes.append([box[0]])
            else:
                rgb_crop = np.array(crop)
                rgb_crop = cv2.resize(rgb_crop, (400, 400))
                car_img = self.transform(rgb_crop).float()
                car_img =  car_img.to(self.device).unsqueeze(0)

                with torch.no_grad():
                    model_pred, make_pred, model_type_pred = self.recognizer(car_img)
                    model_idx = model_pred.argmax(1).item()   
                    make_idx = make_pred.argmax(1).item()
                    model_type_idx = model_type_pred.argmax(1).item()

                    model_name = self.class_names[model_idx]
                    make_name = self.make_names[make_idx]
                    type_name = f"car: {self.model_type_names[model_type_idx]}"
                
                attributes.append([type_name, make_name, model_name])
                

        return crops, attributes
    
    def process_detect_with_attributes(self, rgb_frame):
        detection_results = self.detector.detect(rgb_frame, show=False)
        boxes_with_att = []

        for _, row in detection_results.iterrows():
            if row['name'] == 'ô tô':
                crop_rgb = rgb_frame[int(row['ymin']):int(row['ymax']), int(row['xmin']):int(row['xmax'])].copy()
                crop_rgb = cv2.resize(crop_rgb, (400, 400))
                car_img = self.transform(crop_rgb).float()
                car_img =  car_img.to(self.device).unsqueeze(0)

                with torch.no_grad():
                    _, make_pred, model_type_pred = self.recognizer(car_img)
                    #model_idx = model_pred.argmax(1).item()   
                    make_idx = make_pred.argmax(1).item()
                    model_type_idx = model_type_pred.argmax(1).item()

                    make_name = self.make_names[make_idx]
                    type_name = self.model_type_names[model_type_idx]
                
                boxes_with_att.append((
                    row['name'],
                    (int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])),
                    type_name, 
                    make_name))
            
            else:
                boxes_with_att.append((
                    row['name'],
                    (int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']))
                ))

        return boxes_with_att
    
    def draw_box_with_att(self, rgb_img, boxWithAttList):
        annotated_frame = rgb_img.copy()
        for box_wa in boxWithAttList:
            note = ""
            if box_wa[0] == "ô tô":
                box_color = (0, 0 ,255)
                note = f"{box_wa[2]}, {box_wa[3]}"
            else:
                box_color = (255, 0, 0)
                note = box_wa[0]  

            cv2.rectangle(annotated_frame, (box_wa[1][0], box_wa[1][1]), (box_wa[1][2], box_wa[1][3]), box_color, 2)
            #cv2.putText(annotated_frame, note, (box_wa[1][0], box_wa[1][1] - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

            img_pil = Image.fromarray(annotated_frame)
            draw = ImageDraw.Draw(img_pil)
            draw.text((box_wa[1][0], box_wa[1][1] - self.font_height - 5), note, font=ImageFont.truetype(self.font_path, self.font_height), fill=box_color)
            annotated_frame = np.array(img_pil)

        return annotated_frame

if __name__ == '__main__':
    #attRecognizer = CarMMTRecognizer(arch='resnext50', version=2, weight_path='../pretrained/car_att/resnext50_400_60_v2.pth')
    attRecognizer = CarMMTRecognizer()

    img_path = "test.png"
    video_path = "0531.mp4"
    rtsp_path = "rtsp://admin:IT%40vkist1@10.1.8.227:554/stream1"
    #attRecognizer.recognize_img(img_path=img_path)
    attRecognizer.recognize_video(video_path=rtsp_path)