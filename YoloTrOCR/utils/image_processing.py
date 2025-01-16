import os
import cv2
from PIL import Image
import numpy as np


class PreprocessImage:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def preprocess_image(self, image):
        """
        Preprocesses image into size (224, 224) compatible for ViT
        :param image: ndarray
        :return:
        """
        # Convert OpenCV image (NumPy array) to PIL Image
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Resize while maintaining aspect ratio
        original_size = image.size

        # Calculate the new size while maintaining aspect ratio
        aspect_ratio = original_size[0] / original_size[1]
        if aspect_ratio > 1:  # Width is greater than height
            new_width = self.target_size[0]
            new_height = int(self.target_size[0] / aspect_ratio)
        else:  # Height is greater than width
            new_height = self.target_size[1]
            new_width = int(self.target_size[1] * aspect_ratio)

        # Resize the image
        resized_img = image.resize((new_width, new_height))

        # Calculate padding values
        padding_width = self.target_size[0] - new_width
        padding_height = self.target_size[1] - new_height

        # Apply padding to center the resized image
        pad_left = padding_width // 2
        pad_top = padding_height // 2
        pad_image = Image.new('RGB', self.target_size, (255, 255, 255))  # White background
        pad_image.paste(resized_img, (pad_left, pad_top))

        # Convert back to NumPy array for OpenCV compatibility
        pad_image = cv2.cvtColor(np.array(pad_image), cv2.COLOR_RGB2BGR)

        return pad_image

