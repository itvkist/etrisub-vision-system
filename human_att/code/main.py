import cv2
import logging
import numpy as np
from PIL import Image
from tqdm import tqdm
from processing c

def main(frame, frame_count, path_img_out):
    # Convert frame from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert NumPy array to PIL Image
    pil_image = Image.fromarray(frame_rgb)
    print(type(pil_image))  # Print the type of the image

    # Gọi hàm xử lý các task
    process_frame_tasks(frame_rgb, pil_image, frame_count, path_img_out)


frame_count = 0  
path_img_out = "/home/coder/car/code/human_att/data/"  

frame = cv2.imread("/home/coder/car/data/anh_nguoi.jpg")


main(frame, frame_count, path_img_out)

