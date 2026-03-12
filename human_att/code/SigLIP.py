import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import os
from tqdm.auto import tqdm

import random
from torch.optim import AdamW
import traceback

class InferDataset(Dataset):
    def __init__(self, crop, transform):
            self.crop = crop
            self.transform = transform

    def __getitem__(self, index):
        if index != 0:
            raise IndexError("InferDataset only supports a single item, index must be 0.")
        pixel_values = self.transform(self.crop)
        return pixel_values

def load_model(model_id, id2label, ckpt_path=None, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(
        model_id, problem_type="multi_label_classification", id2label=id2label, ignore_mismatched_sizes=True
        )
    model = model.to(device)

    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model

def infer_image(pixel_values, model, device, threshold=0.5):
    pixel_values = pixel_values.to(device)
    with torch.no_grad():
        outputs = model(pixel_values.unsqueeze(0))  # Thêm batch dimension
        logits = outputs.logits
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float().cpu().numpy()[0]  # [num_labels]
    return preds
