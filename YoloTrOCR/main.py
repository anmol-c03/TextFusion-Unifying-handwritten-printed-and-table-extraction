import os

import sys
import torch
import torchvision
from doclayout_yolo import YOLOv10

import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
# from visualization import visualize_bbox
from text_recognition import TextRecognition
from text_detection import TextDetection
from Doclayout_yolo import *


root_path = os.path.dirname(os.path.abspath(__file__))
OG_IMG_DIR = os.path.join(root_path,  'images', 'original')
RESIZED_IMG_DIR = os.path.join(root_path, 'images', 'resized')
TEXT_FILE_DIR = os.path.join(root_path, 'txt')

model_dir=os.path.join(root_path, "model_doclayout", "DocLayout-YOLO-DocStructBench", "doclayout_yolo_docstructbench_imgsz1024.pt")
img_path=os.path.join(root_path,"aakritti_handwritten.jpg")

doclayout_yolo=Doclayoutyolo(model_dir,img_path)
doclayout_yolo.crop_images()


def pipeline_function(img_file):
    """
    Detect texts on the provided document, resize the and generate respective text and saves into .txt file
    :param img_file: Path of document
    :return:
    """
    text_det_obj = TextDetection(image_file=img_file)
    text_rec_obj = TextRecognition()

    _, cropped_images_file_name = text_det_obj.return_cropped_images()

    texts = []
    for cropped_image in cropped_images_file_name:
        generated_text = text_rec_obj.return_generated_text(image_path=os.path.join(RESIZED_IMG_DIR, cropped_image))
        texts.append(generated_text)


    with open(os.path.join(TEXT_FILE_DIR, "predicted_test.txt"), 'w') as file:
        for text in texts:
            file.write(text)
            file.write(' ')


if __name__ == "__main__":
    pipeline_function(img_file='test3.jpg')