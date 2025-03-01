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
    def return_generated_text(image) :
        """
        Function to return text associated with each cropped image file
        :param image_path: OpenCV image (NumPy array)
        :return: generated_text
        """

        # Convert OpenCV image (NumPy array) to PIL Image
        # pil_image = Image.open(image_path)
        # print(type(image))


        if TextRecognition._processor is None:
            raise ValueError("Processor is not initialized.")

        # Process the image
        pixel_values = TextRecognition._processor(image, return_tensors="pt").pixel_values

        # Move pixel values to the specified device
        pixel_values = pixel_values.to(TextRecognition.device)

        # Generate text
        generated_ids = TextRecognition._model.generate(pixel_values)
        generated_text = TextRecognition._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return generated_text