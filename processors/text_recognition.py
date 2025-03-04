import os
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch


MODEL_NAME = 'microsoft/trocr-large-handwritten'

class TextRecognition:
    _model = None
    _processor = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check for GPU availability

    def __init__(self):


        if TextRecognition._model is None:
            TextRecognition._model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

            TextRecognition._model.to(TextRecognition.device)  # Move the model to the specified device



        if TextRecognition._processor is None:
            TextRecognition._processor = TrOCRProcessor.from_pretrained(MODEL_NAME)

    @staticmethod
    def return_generated_text(images_list):
        """
        Function to process a batch of images at once
        :param images_list: List of OpenCV images (NumPy arrays)
        :return: List of generated text strings in the same order as input images
        """
        # if not images_list:
        #     return []
            
        if TextRecognition._processor is None:
            raise ValueError("Processor is not initialized.")
            
        # Process all images in a single batch
        batch_pixel_values = TextRecognition._processor(images=images_list, return_tensors="pt").pixel_values
        
        # Move pixel values to the specified device
        batch_pixel_values = batch_pixel_values.to(TextRecognition.device)
        
        # Generate all text at once
        batch_generated_ids = TextRecognition._model.generate(batch_pixel_values)
        batch_generated_text = TextRecognition._processor.batch_decode(batch_generated_ids, skip_special_tokens=True)
        print('trocr with yolo')
        
        return batch_generated_text