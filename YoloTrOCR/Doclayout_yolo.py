from doclayout_yolo import YOLOv10
from huggingface_hub import snapshot_download
import torch
import os
import numpy as np
from PIL import Image
import cv2
root_path = os.path.abspath(os.getcwd())
import torchvision

''' TODO: in main.py 
iamge_path,
model_path
'''
model_dir = snapshot_download('juliozhao/DocLayout-YOLO-DocStructBench', local_dir='./model_doclayout/DocLayout-YOLO-DocStructBench')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Doclayoutyolo:
    def __init__(self,
               model_path,
               img_path ):
        self.model =  YOLOv10(model_path)
        self.conf_threshold = 0.25  # Adjust this value as needed
        self.iou_threshold = 0.45
        self.res=None
        self.input_img=cv2.imread(img_path)
        self.id_to_names = {
            0: 'title',
            1: 'plain_text',
            2: 'abandon',
            3: 'figure',
            4: 'figure_caption',
            5: 'table',
            6: 'table_caption',
            7: 'table_footnote',
            8: 'isolate_formula',
            9: 'formula_caption'
            }

    def visualize_bbox(self,
                     boxes, 
                     classes, 
                     scores, 
                     ):
        img = np.array(self.input_img.copy())
        for box, cls, score in zip(boxes, classes, scores):
            x1, y1, x2, y2 = map(int, box)
            class_name = self.id_to_names.get(int(cls), "Unknown")
            label = f"{class_name}: {score:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        return Image.fromarray(img)
    


    def predict(self):
      self.res = self.model.predict(
              self.input_img,
              imgsz=1024,
              device=device,
          )[0]
      boxes = self.res.__dict__['boxes'].xyxy
      classes = self.res.__dict__['boxes'].cls
      scores = self.res.__dict__['boxes'].conf

      indices = torchvision.ops.nms(boxes=torch.Tensor(boxes), scores=torch.Tensor(scores),iou_threshold=self.iou_threshold)
      boxes, scores, classes = boxes[indices], scores[indices], classes[indices]
      if len(boxes.shape) == 1:
          self.boxes = np.expand_dims(boxes, 0)
          scores = np.expand_dims(scores, 0)
          classes = np.expand_dims(classes, 0)
      vis_result=self.visualize_bbox( boxes, classes, scores)
      vis_result.save("bboxes_image.jpg")
      return boxes,classes,scores
    
    @staticmethod
    def apply_filter(cropped_image):
        blurred_image=cv2.GaussianBlur(cropped_image, (5, 5), 0)
        #add padding metrics
        top, bottom, left, right = 20, 20, 20, 20 # Padding sizes
        border_color = (255,255, 255)  # Blue color in BGR
        padded_image = cv2.copyMakeBorder(blurred_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_color)
        return padded_image

    def crop_images(self):
        '''
        Crops the input image based on the predicted bounding boxes and saves the cropped images.

        The function first calls the `predict` method to obtain the predicted bounding boxes, classes, and scores
        It then iterates over each bounding box, crops the corresponding region from the input image, applies a filter
        using the `apply_filter` method, and saves the cropped image with a filename indicating the class and index.

        Parameters:
        None

        Returns:
        None
        '''
        boxes,classes,scores=self.predict()
        for i in range(len(scores)):
            print(self.id_to_names[classes[i].item()])
            if self.id_to_names[classes[i].item()]=='table':
                continue
            x1, y1, x2, y2 = map(int, boxes[i])

            cropped_img = self.input_img[y1:y2, x1:x2]
            img_padded=self.apply_filter(cropped_img)
            cv2.imwrite(f"images/original/{self.id_to_names[classes[i].item()]}_{i}.jpg", img_padded)
