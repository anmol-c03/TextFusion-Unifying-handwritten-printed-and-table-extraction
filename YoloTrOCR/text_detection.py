import os
from typing import List, Tuple, Union
import numpy as np
from ultralytics import YOLO
import cv2
from utils.image_processing import PreprocessImage

# Get the parent directory of the current Python file (handwritten_document_conversion)
root_dir = os.path.dirname(os.path.abspath(__file__))

# Set the correct paths for models and images
MODEL_DIR = os.path.join(root_dir, 'models')
model_path = os.path.join(MODEL_DIR, 'best_line.pt')
OG_IMG_DIR = os.path.join(root_dir, 'images', 'original')
RESIZED_IMG_DIR = os.path.join(root_dir, 'images', 'resized')

class TextDetection:
    _model = None

    def __init__(self, image_file: str) -> None:
        self.image_file = image_file
        # self.image_preprocess = PreprocessImage()

        if TextDetection._model is None:
            TextDetection._model = YOLO(model_path)

    def detect(self) -> list:
        """
        Function to return results from the TextDetection Model.
        """
        return TextDetection._model(os.path.join(OG_IMG_DIR, self.image_file))

    def calculate_dynamic_thresholds(self, image: np.ndarray) -> Tuple[int, int]:
        """
        Calculate dynamic thresholds based on the image size.
        :param image: Input image to base the threshold calculation.
        :return: Dynamic line_threshold and column_threshold.
        """
        # Image dimensions
        image_height, image_width = image.shape[:2]

        # Dynamic thresholds as a percentage of image height/width
        line_threshold = int(image_height * 0.05)  # 5% of image height
        column_threshold = int(image_width * 0.02)  # 2% of image width

        return line_threshold, column_threshold

    @staticmethod
    def group_bboxes_by_lines(
            bboxes_with_centers: List[Tuple[list[int], Tuple[int, int]]],
        line_threshold: int
    ) -> List[List[Tuple[list[int], Tuple[int, int]]]]:
        """
        Group bounding boxes into lines based on their vertical proximity (y-coordinates).
        :param bboxes_with_centers: List of bounding boxes with their centers.
        :param line_threshold: Dynamic threshold for grouping boxes into lines.
        :return: List of lines, each line being a list of bounding boxes.
        """
        # Sort bounding boxes by y-coordinate (top to bottom)
        bboxes_with_centers.sort(key=lambda x: x[1][1])

        lines = []
        current_line = []

        for bbox_with_center in bboxes_with_centers:
            bbox, center = bbox_with_center
            if not current_line:
                current_line.append(bbox_with_center)
            else:
                last_center_y = current_line[-1][1][1]
                if abs(last_center_y - center[1]) <= line_threshold:
                    current_line.append(bbox_with_center)
                else:
                    lines.append(current_line)
                    current_line = [bbox_with_center]

        if current_line:
            lines.append(current_line)

        return lines

    def sort_bboxes_linewise_and_columnwise(
        self,
        bboxes_with_centers: List[Tuple[list[int], Tuple[int, int]]],
        line_threshold: int,
        column_threshold: int
    ) -> List[Union[Tuple[list[int], Tuple[int, int]], str]]:
        """
        Sort bounding boxes first line-wise (group by y-coordinate proximity),
        then column-wise (sort by x-coordinate within each line).
        Include line separators between each line.
        :param bboxes_with_centers: List of bounding boxes with their centers.
        :param line_threshold: Dynamic threshold for grouping boxes into lines.
        :param column_threshold: Dynamic threshold for sorting columns.
        :return: Sorted list of bounding boxes with line separators.
        """
        # Group bounding boxes by lines (vertical alignment)
        lines = self.group_bboxes_by_lines(bboxes_with_centers, line_threshold)

        sorted_bboxes_with_separators = []
        for idx, line in enumerate(lines):
            if idx > 0:
                # Insert a line separator between lines
                sorted_bboxes_with_separators.append('\n')  # Or '\n' for text separation
            # Sort each line by x-coordinate (left to right)
            sorted_line = sorted(line, key=lambda x: x[1][0])
            sorted_bboxes_with_separators.extend(sorted_line)

        return sorted_bboxes_with_separators

    def return_bboxes(self) -> List[List[int]]:
        """
        Function to return bounding boxes of the detected text.
        :return: List of bounding boxes.
        """
        results = self.detect()
        bboxes = [
            [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
            for result in results
            for box in result.boxes.data.tolist()
        ]
        return bboxes

    def return_cropped_images(self) -> Tuple[List[np.ndarray], List[str]]:
        """
        Function to return a list of cropped images and their file names.
        :return: List of cropped images and file names.
        """
        image = cv2.imread(os.path.join(OG_IMG_DIR, self.image_file))
        bboxes = self.return_bboxes()

        # Calculate centers of the bounding boxes
        bboxes_with_centers = [
            (bbox, ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2))
            for bbox in bboxes
        ]

        # Get dynamic thresholds based on image size
        line_threshold, column_threshold = self.calculate_dynamic_thresholds(image)

        # Sort bounding boxes line-wise and column-wise, including line separators
        sorted_bboxes_with_centers = self.sort_bboxes_linewise_and_columnwise(
            bboxes_with_centers, line_threshold, column_threshold
        )

        cropped_images = []
        cropped_images_file_name = []

        # Crop images based on sorted bounding boxes
        for idx, item in enumerate(sorted_bboxes_with_centers):
            if isinstance(item, str):
                # Handle line separator (e.g., a marker or placeholder)
                print(item)  # Optionally print the separator for debugging
                continue  # Skip processing for line separators

            bbox, _ = item
            x1, y1, x2, y2 = bbox
            cropped_image = image[y1:y2, x1:x2]
            # cropped_image = self.image_preprocess.preprocess_image(cropped_image)
            # cropped_image = cv2.resize(cropped_image, (224, 224))
            cropped_images.append(cropped_image)

            file_name = f"{os.path.splitext(self.image_file)[0]}_{idx + 1}{os.path.splitext(self.image_file)[-1]}"
            cropped_images_file_name.append(file_name)

            cv2.imwrite(os.path.join(RESIZED_IMG_DIR, file_name), cropped_image)

        return cropped_images, cropped_images_file_name
    
if __name__ == '__main__':
    image_file = 'sample1.jpg'
    text_detection = TextDetection('test1.jpeg')
    cropped_images, cropped_images_file_name = text_detection.return_cropped_images()
    print(cropped_images_file_name)