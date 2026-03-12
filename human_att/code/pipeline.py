import cv2
import torch 
import numpy as np
from PIL import Image
from ultralytics import YOLO
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from human_att.code.SigLIP import InferDataset, load_model, infer_image

transform = Compose([
    Resize((384, 384)),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

class SigLIP:
    def __init__(self, model_name="google/siglip-so400m-patch14-384"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.detection_model = YOLO("pretrained/yolo11m.pt")
        self.torch_dtype = torch.float32
        self.conf_threshold = 0.2
        self.id2label = {0: 'accessoryHeadphone', 1: 'personalLess15', 2: 'personalLess30', 3: 'personalLess45', 4: 'personalLess60', 5: 'personalLarger60', 6: 'carryingBackpack', 7: 'hairBald', 8: 'footwearBoots', 9: 'lowerBodyCapri', 10: 'carryingOther', 11: 'carryingShoppingTro', 12: 'carryingUmbrella', 13: 'lowerBodyCasual', 14: 'upperBodyCasual', 15: 'personalFemale', 16: 'carryingFolder', 17: 'lowerBodyFormal', 18: 'upperBodyFormal', 19: 'accessoryHairBand', 20: 'accessoryHat', 21: 'lowerBodyHotPants', 22: 'upperBodyJacket', 23: 'lowerBodyJeans', 24: 'accessoryKerchief', 25: 'footwearLeatherShoes', 26: 'upperBodyLogo', 27: 'hairLong', 28: 'lowerBodyLongSkirt', 29: 'upperBodyLongSleeve', 30: 'lowerBodyPlaid', 31: 'lowerBodyThinStripes', 32: 'carryingLuggageCase', 33: 'personalMale', 34: 'carryingMessengerBag', 35: 'accessoryMuffler', 36: 'accessoryNothing', 37: 'carryingNothing', 38: 'upperBodyNoSleeve', 39: 'upperBodyPlaid', 40: 'carryingPlasticBags', 41: 'footwearSandals', 42: 'footwearShoes', 43: 'hairShort', 44: 'lowerBodyShorts', 45: 'upperBodyShortSleeve', 46: 'lowerBodyShortSkirt', 47: 'footwearSneaker', 48: 'footwearStocking', 49: 'upperBodyThinStripes', 50: 'upperBodySuit', 51: 'carryingSuitcase', 52: 'lowerBodySuits', 53: 'accessorySunglasses', 54: 'upperBodySweater', 55: 'upperBodyThickStripes', 56: 'lowerBodyTrousers', 57: 'upperBodyTshirt', 58: 'upperBodyOther', 59: 'upperBodyVNeck', 60: 'footwearBlack', 61: 'footwearBlue', 62: 'footwearBrown', 63: 'footwearGreen', 64: 'footwearGrey', 65: 'footwearOrange', 66: 'footwearPink', 67: 'footwearPurple', 68: 'footwearRed', 69: 'footwearWhite', 70: 'footwearYellow', 71: 'hairBlack', 72: 'hairBlue', 73: 'hairBrown', 74: 'hairGreen', 75: 'hairGrey', 76: 'hairOrange', 77: 'hairPink', 78: 'hairPurple', 79: 'hairRed', 80: 'hairWhite', 81: 'hairYellow', 82: 'lowerBodyBlack', 83: 'lowerBodyBlue', 84: 'lowerBodyBrown', 85: 'lowerBodyGreen', 86: 'lowerBodyGrey', 87: 'lowerBodyOrange', 88: 'lowerBodyPink', 89: 'lowerBodyPurple', 90: 'lowerBodyRed', 91: 'lowerBodyWhite', 92: 'lowerBodyYellow', 93: 'upperBodyBlack', 94: 'upperBodyBlue', 95: 'upperBodyBrown', 96: 'upperBodyGreen', 97: 'upperBodyGrey', 98: 'upperBodyOrange', 99: 'upperBodyPink', 100: 'upperBodyPurple', 101: 'upperBodyRed', 102: 'upperBodyWhite', 103: 'upperBodyYellow'}
        self.ckpt_path = "pretrained/SigLIP_finetuned.pth"
        self.model = load_model(model_name, self.id2label, self.ckpt_path, self.device)
        

    def generate(self, task_prompt, image, text_input=None):
        prompt = task_prompt if text_input is None else task_prompt + text_input

        inputs = self.processor(text=prompt, images=image, return_tensors="pt")

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"].to(self.device),
            pixel_values=inputs["pixel_values"].to(self.device),
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False)[0]

        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )

        return parsed_answer

    def get_boxes_by_label(self, data, label):
        boxs = []
        for i, lbl in enumerate(data['<OD>']['labels']):
            if lbl == label:
                bbox = data['<OD>']['bboxes'][i]
                boxs.append(bbox)
        return boxs

    def draw_boxes(self, image, boxes, color=(0, 255, 0), thickness=2):
        for box in boxes:
            start_point = (int(box[0]), int(box[1]))
            end_point = (int(box[2]), int(box[3]))
            image = cv2.rectangle(image, start_point,
                                  end_point, color, thickness)
        return image

    def process_frame_tasks(self, frame_rgb, path_img_out="output_directory/"):
        detection_results = self.detection_model(frame_rgb, verbose=False)[0]
        annotated_frame = frame_rgb.copy()
        caption_results = []
        image_results = []
        
        # Generate caption
        pil_image = Image.fromarray(frame_rgb)
        # Object detection
        for result in detection_results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if class_id in [0]:
                if score > self.conf_threshold:
                    person_img = frame_rgb[int(y1):int(y2), int(x1):int(x2)]
                    crop_area = [x1, y1, x2, y2]
                    crop = pil_image.crop(crop_area)
                    dataset = InferDataset(crop, transform=transform)
                    pixel_values = dataset[0]
                    pred = infer_image(pixel_values, self.model, self.device, threshold=0.8)
                    caption = [self.id2label[j] for j, p in enumerate(pred) if p == 1]
                    caption_dict = {"<DETAILED_CAPTION>": ", ".join(caption)}
                    cv2.rectangle(annotated_frame, (int(x1), int(
                        y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    caption_results.append(caption_dict)
                    image_results.append(person_img)
        return annotated_frame, caption_results, caption_results, image_results

    def process_detect(self, frame_rgb, path_img_out="output_directory/"):
        detection_results = self.detection_model(frame_rgb,verbose = False)[0]
        annotated_frame = frame_rgb.copy()
        crop_area_list = []
        # Object detection
        for result in detection_results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                if class_id in [0]:
                    if score > self.conf_threshold:
                        crop_area = [x1, y1, x2, y2]

                        crop_area_list.append(crop_area)
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        return annotated_frame,crop_area_list

# Usage example:
# florence2_model = SigLIP()
# frame_rgb = cv2.imread("human_att/code/testimg.jpg") # Load your frame
# pil_image = Image.fromarray(cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB))
# processed_frame = florence2_model.process_frame_tasks(frame_rgb)
# processed_frame = florence2_model.process_frame_tasks(frame_rgb)
