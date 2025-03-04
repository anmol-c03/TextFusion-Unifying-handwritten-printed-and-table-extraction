import os
import numpy as np
import cv2
import torch
import torchvision
from PIL import Image
from doclayout_yolo import YOLOv10
from huggingface_hub import snapshot_download

root_path = os.path.abspath(os.getcwd())
model_dir = snapshot_download('juliozhao/DocLayout-YOLO-DocStructBench', local_dir='./model_doclayout/DocLayout-YOLO-DocStructBench')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class LayoutProcessor:
    def __init__(self, model_path, img_path):
        self.model = YOLOv10(model_path)
        self.conf_threshold = 0.05 # Lower confidence threshold to detect low confidence detections
        self.iou_threshold = 0.1  # IOU threshold for NMS
        self.res = None
        self.input_img = cv2.imread(img_path)
        self.id_to_names = {
            0: 'Title',
            1: 'PlainText',
            2: 'Abandon',
            3: 'Figure',
            4: 'FigureCaption',
            5: 'Table',
            6: 'TableCaption',
            7: 'TableFootnote',
            8: 'IsolateFormula',
            9: 'IormulaCaption'
        }
        
        # Create output directory if it doesn't exist
        os.makedirs("images/original", exist_ok=True)
        

    def predict(self):
        """
        Run model prediction and apply NMS to filter overlapping boxes
        """
        self.res = self.model.predict(
            self.input_img,
            imgsz=1024,
            device=device,
            conf=self.conf_threshold
        )[0]
        
        boxes = self.res.__dict__['boxes'].xyxy
        classes = self.res.__dict__['boxes'].cls
        scores = self.res.__dict__['boxes'].conf

        # Apply standard NMS
        indices = torchvision.ops.nms(boxes=torch.Tensor(boxes), 
                                      scores=torch.Tensor(scores), 
                                      iou_threshold=self.iou_threshold)
        
        boxes, scores, classes = boxes[indices], scores[indices], classes[indices]
        
        # Ensure we have proper dimensions
        if len(boxes) == 0:
            return np.array([]), np.array([]), np.array([])
            
        if len(boxes.shape) == 1:
            boxes = np.expand_dims(boxes, 0)
            scores = np.expand_dims(scores, 0)
            classes = np.expand_dims(classes, 0)
        
        return boxes, classes, scores

    def is_contained_within(self, box1, box2):
        """
        Check if box1 is completely contained within box2
        Returns True if box1 is inside box2, False otherwise
        
        Args:
            box1: [x1, y1, x2, y2] - box that might be contained
            box2: [x1, y1, x2, y2] - potentially containing box
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Check if box1 is completely inside box2
        if (x1_1 >= x1_2 and x2_1 <= x2_2 and 
            y1_1 >= y1_2 and y2_1 <= y2_2):
            return True
        return False
    
    def calculate_box_area(self, box):
        """Calculate the area of a box"""
        x1, y1, x2, y2 = box
        return (x2 - x1) * (y2 - y1)

    def filter_contained_boxes(self, boxes, classes, scores):
        """
        Remove smaller bounding boxes that are completely contained within larger ones
        """
        if len(boxes) <= 1:
            return boxes, classes, scores
        
        # Create a list of all detections with their areas
        detections = []
        for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
            area = self.calculate_box_area(box)
            detections.append((i, box, cls, score, area))
        
        # Sort by area (largest first) to prioritize larger boxes
        detections.sort(key=lambda x: x[4], reverse=True)
        
        # Track which indices to keep
        indices_to_remove = set()
        
        # For each box, check if smaller boxes are contained within it
        for i in range(len(detections)):
            if i in indices_to_remove:
                continue
                
            box_i = detections[i][1]
            
            for j in range(i+1, len(detections)):
                if j in indices_to_remove:
                    continue
                    
                box_j = detections[j][1]
                
                # If box j is completely contained in box i, mark it for removal
                if self.is_contained_within(box_j, box_i):
                    indices_to_remove.add(j)
        
        # Create filtered lists
        filtered_boxes = []
        filtered_classes = []
        filtered_scores = []
        
        for i, (idx, box, cls, score, _) in enumerate(detections):
            if i not in indices_to_remove:
                filtered_boxes.append(box.cpu())
                filtered_classes.append(cls.cpu())
                filtered_scores.append(score.cpu())
        
        return np.array(filtered_boxes), np.array(filtered_classes), np.array(filtered_scores)

    @staticmethod
    def apply_filter(cropped_image):
        """
        Apply filters to enhance the cropped image
        """
        blurred_image = cv2.GaussianBlur(cropped_image, (5, 5), 0)
        top, bottom, left, right = 20, 20, 20, 20
        border_color = (255, 255, 255)
        padded_image = cv2.copyMakeBorder(blurred_image, top, bottom, left, right, 
                                          cv2.BORDER_CONSTANT, value=border_color)
        return padded_image

    def crop_images(self):
        """
        Crops the input image based on predicted bounding boxes, sorted from top to bottom.
        Applies containment filtering to remove boxes completely inside others.
        Saves cropped images with filenames indicating their class and index.
        """
        boxes, classes, scores = self.predict()
        
        # Apply containment filtering instead of IoU filtering
        boxes, classes, scores = self.filter_contained_boxes(boxes, classes, scores)
        
        if len(boxes) == 0:
            print("No boxes detected.")
            return
        
        padding = 10
        
        # Create a list of tuples (box, class, score) and sort by y1 coordinate (top of the box)
        sorted_detections = sorted(
            [(box, cls, score) for box, cls, score in zip(boxes, classes, scores)],
            key=lambda x: x[0][1]  # Sort by y1 coordinate
        )
        
        # Filter and save images in sorted order
        for i, (box, cls, score) in enumerate(sorted_detections):
            class_name = self.id_to_names[cls.item()]
            
            # Skip tables and abandoned sections
            if class_name == 'Table' or class_name == 'Abandon':
                continue
            
            x1, y1, x2, y2 = map(int, box)
            
            # Add padding but stay within image boundaries
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(self.input_img.shape[1], x2 + padding)
            y2 = min(self.input_img.shape[0], y2 + padding)
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue
                
            cropped_img = self.input_img[y1:y2, x1:x2]
            
            # Skip empty images
            if cropped_img.size == 0:
                continue
                
            img_padded = self.apply_filter(cropped_img)
            
            # Save with ordered index to maintain sorting
            output_path = f"./images/original/{class_name}_{i+1:03d}.jpg"
            cv2.imwrite(output_path, cropped_img)
            print(f"Saved: {output_path}")

    def visualize_bbox(self):
        """
        Visualize bounding boxes on the image with class labels and confidence scores
        """
        boxes, classes, scores = self.predict()
        
        # Apply containment filtering
        boxes, classes, scores = self.filter_contained_boxes(boxes, classes, scores)
        
        img = np.array(self.input_img.copy())
        
        for box, cls, score in zip(boxes, classes, scores):
            x1, y1, x2, y2 = map(int, box)
            class_name = self.id_to_names.get(int(cls), "Unknown")
            label = f"{class_name}: {score:.2f}"
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.imwrite('visualization_bbox_doclayout.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        

