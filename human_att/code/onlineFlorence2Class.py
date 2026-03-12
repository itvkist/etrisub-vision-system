import torch
#from transformers import AutoProcessor, AutoModelForCausalLM
import cv2
from PIL import Image
import os
from ultralytics import YOLO
import requests
import io, base64

ngrok_url = "https://bb67c9762709.ngrok-free.app/generate"


def xywh_to_x1y1x2y2(x, y, w, h):
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return x1, y1, x2, y2

    
class Florence2Model:
    def __init__(self, model_name="microsoft/Florence-2-base"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.detection_model = model = YOLO("pretrained/yolov8n.pt")
        self.torch_dtype = torch.float32
        self.conf_threshold = 0.4
        #self.model = AutoModelForCausalLM.from_pretrained(
        #    model_name,
        #    torch_dtype=self.torch_dtype,
        #    trust_remote_code=True
        #).to(self.device)
        
        #self.processor = AutoProcessor.from_pretrained(
        #    model_name,
        #    trust_remote_code=True
        #)

    def generate(self, task_prompt, image, text_input=None):

        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        img_64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        payload = {
            "task_prompt": task_prompt,
            "image_b64": img_64,
            "text_input": text_input
        }

        res = requests.post(url=ngrok_url, json=payload)
        # still synchronous call
        return res.json()

        #prompt = task_prompt if text_input is None else task_prompt + text_input
        
        #inputs = self.processor(text=prompt, images=image, return_tensors="pt")

        #generated_ids = self.model.generate(
        #    input_ids=inputs["input_ids"].to(self.device),
        #    pixel_values=inputs["pixel_values"].to(self.device),
        #    max_new_tokens=1024,
        #    early_stopping=False,
        #    do_sample=False,
        #    num_beams=3,
        #)

        #generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        #parsed_answer = self.processor.post_process_generation(
        #    generated_text,
        #    task=task_prompt,
        #    image_size=(image.width, image.height)
        #)

        #return parsed_answer

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
            image = cv2.rectangle(image, start_point, end_point, color, thickness)
        return image

    def process_frame_tasks(self, frame_rgb, path_img_out="output_directory/"):
        detection_results = self.detection_model(frame_rgb,verbose = False)[0]
        annotated_frame = frame_rgb.copy()
        # Generate caption
        pil_image = Image.fromarray(frame_rgb)
        task_prompt = '<CAPTION>'
        image_results = []
        results = self.generate(task_prompt, pil_image)
        caption_results = []
        # Object detection
        for result in detection_results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                if class_id in [0]:
                    if score > self.conf_threshold:
                        # print(x1, y1, x2, y2)
                        # Extract person image
                        person_img = frame_rgb[int(y1):int(y2), int(x1):int(x2)]
                        # x1, y1, x2, y2 = xywh_to_x1y1x2y2(x1, y1, x2, y2)
                        crop_area = [x1, y1, x2, y2]
                        crop = pil_image.crop(crop_area)
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        task_prompt1 = '<DETAILED_CAPTION>'
                        result1 = self.generate(task_prompt1, crop)

                        # Save cropped image
                        # image_filename = os.path.join(path_img_out, f"{frame_count}_{i}.jpg")
                        # crop.save(image_filename)

                        # Save caption to a text file
                        # caption_filename = os.path.join(path_img_out, f"{frame_count}_{i}.txt")
                        # with open(caption_filename, 'w') as file:
                        #     file.write(result1['<DETAILED_CAPTION>'])
                        # print(result1)
                        caption_results.append(result1)
                        image_results.append(person_img)
        return annotated_frame,results, caption_results, image_results

    def process_detect(self, frame_rgb, path_img_out="output_directory/"):
        detection_results = self.detection_model(frame_rgb,verbose = False)[0]
        annotated_frame = frame_rgb.copy()
        # Generate caption
        # pil_image = Image.fromarray(frame_rgb)
        # task_prompt = '<CAPTION>'
        # results = self.generate(task_prompt, pil_image)
        crop_area_list = []
        # Object detection
        for result in detection_results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                if class_id in [0]:
                    if score > self.conf_threshold:
                        # print(x1, y1, x2, y2)
                        
                        # Extract person image
                        # person_img = frame[int(y1):int(y2), int(x1):int(x2)]
                        # x1, y1, x2, y2 = xywh_to_x1y1x2y2(x1, y1, x2, y2)
                        crop_area = [x1, y1, x2, y2]
                        # crop = pil_image.crop(crop_area)
                        crop_area_list.append(crop_area)
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        # task_prompt1 = '<DETAILED_CAPTION>'
                        # result1 = self.generate(task_prompt1, crop)

                        # # Save cropped image
                        # # image_filename = os.path.join(path_img_out, f"{frame_count}_{i}.jpg")
                        # # crop.save(image_filename)

                        # # Save caption to a text file
                        # # caption_filename = os.path.join(path_img_out, f"{frame_count}_{i}.txt")
                        # # with open(caption_filename, 'w') as file:
                        # #     file.write(result1['<DETAILED_CAPTION>'])
                        # print(result1)
                        # caption_results.append(result1)
        return annotated_frame,crop_area_list

    def process_attribute(self, frame_rgb,crop_area_list, path_img_out="output_directory/"):
        pil_image = Image.fromarray(frame_rgb)
        caption_results = []

        for crop_area in crop_area_list:
            crop_img = pil_image.crop(crop_area)
            task_prompt1 = '<DETAILED_CAPTION>'
            result1 = self.generate(task_prompt1, crop_area)
            caption_results.append(result1)
        print(f"caption_results: {caption_results}")
        return caption_results
        # result_detect = self.generate(task_prompt_detect, pil_image)

        # Find object box
        # label_obj = "person"
        # obj_box = self.get_boxes_by_label(result_detect, label_obj)
        # print(f"Number of detected persons: {len(obj_box)}")
        
        # draw_frame = self.draw_boxes(frame_rgb, obj_box)
        
        # if len(obj_box) > 0:
        #     for i, box in enumerate(obj_box):
        #         # Crop objects
        #         crop_area = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        #         crop = pil_image.crop(crop_area)

        #         task_prompt1 = '<DETAILED_CAPTION>'
        #         result1 = self.generate(task_prompt1, crop)

        #         # Save cropped image
        #         image_filename = os.path.join(path_img_out, f"{frame_count}_{i}.jpg")
        #         crop.save(image_filename)

                # Save caption to a text file
                # caption_filename = os.path.join(path_img_out, f"{frame_count}_{i}.txt")
                # with open(caption_filename, 'w') as file:
                #     file.write(result1['<DETAILED_CAPTION>'])
                # print(result1)

        # Save caption of the whole frame to a text file
        # caption_filename = os.path.join(path_img_out, f"{frame_count}.txt")
        # with open(caption_filename, 'w') as file:
        #     file.write(results['<CAPTION>'])

        # Save the original frame (BGR format)
        # image_filename = os.path.join(path_img_out, f"{frame_count}.jpg")
        # cv2.imwrite(image_filename, cv2.cvtColor(draw_frame, cv2.COLOR_RGB2BGR))
        # print(f"Saved image and caption for frame {frame_count}")

        

# Usage example:
# florence2_model = Florence2Model()
# frame_rgb = ... # Load your frame
# pil_image = Image.fromarray(cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB))
# frame_count = 1
# path_img_out = "output_directory/"
# processed_frame = florence2_model.process_frame_tasks(frame_rgb, pil_image, frame_count, path_img_out)