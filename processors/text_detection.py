import os
from typing import List, Tuple
import numpy as np
from ultralytics import YOLO
import cv2



# Get the parent directory of the current Python file
# Set the correct paths for models and images
MODEL_DIR = os.path.join('.', 'models')
model_path = os.path.join(MODEL_DIR, 'best_line.pt')
OG_IMG_DIR = os.path.join('.', 'images', 'original')
RESIZED_IMG_DIR = os.path.join('.', 'images', 'resized')
VISUALIZATION_DIR = os.path.join('.', 'images', 'visualization')

# Ensure visualization directory exists

class TextDetection:
    _model = None

    def __init__(self, image_file: str, confidence_threshold: float = 0.5, overlap_threshold: float = 0.5) -> None:
        self.image_file = image_file
        self.confidence_threshold = confidence_threshold
        self.overlap_threshold = overlap_threshold

        if TextDetection._model is None:

            TextDetection._model = YOLO(model_path)

    def calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        :param box1: First bounding box [x1, y1, x2, y2]
        :param box2: Second bounding box [x1, y1, x2, y2]
        :return: IoU score
        """
        # Compute coordinates of intersection rectangle
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # Compute area of intersection
        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        # Compute areas of both boxes
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # Compute union
        union = area1 + area2 - intersection

        # Compute IoU
        return intersection / union if union > 0 else 0

    def filter_overlapping_bboxes(self, bboxes: List[List[int]], confidences: List[float]) -> Tuple[List[List[int]], List[float]]:
        """
        Filter out overlapping bounding boxes based on IoU and confidence.
        
        :param bboxes: List of bounding boxes
        :param confidences: Corresponding confidence scores
        :return: Filtered bounding boxes and confidences
        """
        if not bboxes:
            return [], []

        # Sort bboxes by confidence in descending order
        sorted_indices = sorted(range(len(confidences)), key=lambda k: confidences[k], reverse=True)
        filtered_bboxes = []
        filtered_confidences = []

        for idx in sorted_indices:
            current_bbox = bboxes[idx]
            current_confidence = confidences[idx]
            
            # Check if this bbox overlaps significantly with any already selected bbox
            if all(self.calculate_iou(current_bbox, existing_bbox) < self.overlap_threshold 
                   for existing_bbox in filtered_bboxes):
                filtered_bboxes.append(current_bbox)
                filtered_confidences.append(current_confidence)

        return filtered_bboxes, filtered_confidences

    def return_bboxes(self) -> List[List[int]]:
        """
        Function to return bounding boxes of the detected text with confidence filtering.
        :return: List of bounding boxes
        """
        results = self.detect()
        
        # Extract bboxes, confidences, and apply filtering
        bboxes_with_data = [
            (
                [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                float(conf)
            )
            for result in results
            for box, conf in zip(result.boxes.data.tolist(), result.boxes.conf.tolist())
            if float(conf) >= self.confidence_threshold
        ]

        # Separate bboxes and confidences
        bboxes = [bbox for bbox, _ in bboxes_with_data]
        confidences = [conf for _, conf in bboxes_with_data]

        # Filter overlapping bboxes
        filtered_bboxes, _ = self.filter_overlapping_bboxes(bboxes, confidences)

        return filtered_bboxes

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

    def reading_order_sort(self, bboxes_with_centers: List[Tuple[list[int], Tuple[int, int]]]) -> List[Tuple[list[int], Tuple[int, int]]]:
        """
        Sort bounding boxes in a natural reading order: top-to-bottom and left-to-right.
        :param bboxes_with_centers: List of bounding boxes with their centers.
        :return: Sorted list of bounding boxes.
        """
        if not bboxes_with_centers:
            return []

        # Load the image to get its dimensions
        image_path = os.path.join(OG_IMG_DIR, self.image_file)
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        # Sort bounding boxes by y-coordinate first
        bboxes_with_centers.sort(key=lambda item: item[1][1])  # Sort by center Y

        # Group bounding boxes into rows
        grouped_rows = []
        current_row = [bboxes_with_centers[0]]

        for i in range(1, len(bboxes_with_centers)):
            prev_bbox, prev_center = current_row[-1]
            current_bbox, current_center = bboxes_with_centers[i]

            # Check if the current box is on the same line as the previous box
            row_height_threshold = max((prev_bbox[3] - prev_bbox[1]) * 0.7, 10)
            if abs(current_center[1] - prev_center[1]) < row_height_threshold:
                current_row.append(bboxes_with_centers[i])
            else:
                # Sort the current row left to right
                grouped_rows.append(sorted(current_row, key=lambda item: item[1][0]))  # Sort by X
                current_row = [bboxes_with_centers[i]]

        # Append the last row
        grouped_rows.append(sorted(current_row, key=lambda item: item[1][0]))

        # Flatten list back to sorted order
        sorted_bboxes = [bbox for row in grouped_rows for bbox in row]

        # Visualize the sorted bounding boxes
        self.visualize_sorted_boxes(sorted_bboxes, width, height)

        return sorted_bboxes



    def visualize_sorted_boxes(self, sorted_boxes, width, height):
        """
        Create a visualization of the sorted boxes to verify the reading order.
        """
        image_path = os.path.join(OG_IMG_DIR, self.image_file)
        image = cv2.imread(image_path)
        viz_image = image.copy()
        
        # Define colors for visualization
        colors = [
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Cyan
        ]
        
        # Draw boxes in order
        for i, (bbox, center) in enumerate(sorted_boxes):
            color = colors[i % len(colors)]
            cv2.rectangle(viz_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw index number
            cv2.putText(viz_image, str(i+1), (bbox[0], bbox[1]-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw center point
            cv2.circle(viz_image, center, 3, color, -1)
        
        # Save visualization
        viz_path = os.path.join(VISUALIZATION_DIR, f"reading_order_{os.path.basename(self.image_file)}")
        cv2.imwrite(viz_path, viz_image)
        print(f"Reading order visualization saved to: {viz_path}")

    def process_form_structure(self, bboxes_with_centers: List[Tuple[list[int], Tuple[int, int]]]) -> List[Tuple[list[int], Tuple[int, int]]]:
        """
        Process bounding boxes recognizing the form-like structure where related elements
        should be processed together.
        
        :param bboxes_with_centers: List of bounding boxes with their centers.
        :return: List of bounding boxes sorted in a logical reading order for forms.
        """
        # Use the improved reading order sort
        return self.reading_order_sort(bboxes_with_centers)

    def return_cropped_images(self) -> Tuple[List[np.ndarray], List[str]]:
        """
        Function to return a list of cropped images and their file names.
        :return: List of cropped images and file names.
        """
        image_path =  self.image_file
        image = cv2.imread(image_path)
        bboxes = self.return_bboxes()

        # Calculate centers of the bounding boxes
        bboxes_with_centers = [
            (bbox, ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2))
            for bbox in bboxes
        ]

        # Process the form structure to get correctly ordered boxes
        sorted_bboxes_with_centers = self.process_form_structure(bboxes_with_centers)

        # Create debug image with processing sequence
        debug_image = image.copy()
        for i, (bbox, center) in enumerate(sorted_bboxes_with_centers):
            x1, y1, x2, y2 = bbox
            # Draw rectangle with sequence number
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(debug_image, str(i+1), (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Save debug image

        debug_path = os.path.join(VISUALIZATION_DIR, f"debug_{os.path.basename(self.image_file)}")
        cv2.imwrite(debug_path, debug_image)
        print(f"Debug image with processing sequence saved to: {debug_path}")

        cropped_images = []
        cropped_images_file_name = []

        # Crop images based on sorted bounding boxes
        for idx, (bbox, _) in enumerate(sorted_bboxes_with_centers):
            x1, y1, x2, y2 = bbox
            cropped_image = image[y1:y2, x1:x2]
            cropped_images.append(cropped_image)

            file_name = f"{os.path.basename(self.image_file).split('.')[0]}_{idx + 1}{os.path.splitext(self.image_file)[-1]}"
            cropped_images_file_name.append(file_name)
            
            output_path = os.path.join(RESIZED_IMG_DIR, file_name)
            print(f"Saving cropped image {idx+1}: {output_path}")
            cv2.imwrite(output_path, cropped_image)

        return cropped_images, cropped_images_file_name