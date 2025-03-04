from transformers import AutoModelForObjectDetection
import torch
from PIL import Image
from torchvision import transforms
from .preprocess import *
from .crop_table import objects_to_crops
from transformers import TableTransformerForObjectDetection
from .cell_coordinates import get_cell_coordinates_by_row
from .ocr import Recognize
import csv


def extract(ocr,img_path):

    model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection", revision="no_timm")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print(img_path)
    image = Image.open(img_path).convert("RGB")
    # let's display it a bit smaller
    width, height = image.size

    detection_transform = transforms.Compose([
            MaxResize(800),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    pixel_values = detection_transform(image).unsqueeze(0)
    pixel_values = pixel_values.to(device)


    with torch.no_grad():
        outputs = model(pixel_values)

    id2label = model.config.id2label
    id2label[len(model.config.id2label)] = "no object"

    objects = outputs_to_objects(outputs, image.size, id2label)

    
    tokens = []
    detection_class_thresholds = {
        "table": 0.5,
        "table rotated": 0.5,
        "no object": 10
    }
    crop_padding = 10
    cropped_table = []
    tables_crops = objects_to_crops(image, tokens, objects, detection_class_thresholds, padding=0)
    try:
        for i in range(0, len(tables_crops)):
            cropped_table.extend([tables_crops[i]['image'].convert("RGB")])
    except Exception as e:
        print("Error cropping tables:", e)

    structure_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-structure-recognition-v1.1-all")
    structure_model.to(device)
    outputs,cells = [],[]

    structure_transform = transforms.Compose([
        MaxResize(1000),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    structure_id2label = structure_model.config.id2label
    structure_id2label[len(structure_id2label)] = "no object"

    for crop in cropped_table:
        pixel_values = structure_transform(crop).unsqueeze(0)
        pixel_values = pixel_values.to(device)
        with torch.no_grad():
            output = structure_model(pixel_values)
            outputs.append(output)

    for i in range(len(cropped_table)):
        cell = outputs_to_objects(outputs[i], cropped_table[i].size, structure_id2label)
        cells.extend([cell])
    
    cell_coordinates = []
    structured_data=[]
    for cell in cells:
        cell_coordinate = get_cell_coordinates_by_row(cell)
        cell_coordinates.extend([cell_coordinate])
    
    paddle_ocr=Recognize(ocr)
    for i in range(len(cell_coordinates)):
        data = paddle_ocr.apply_ocr(cell_coordinates[i],cropped_table[i])
        structured_data.extend([data])

    final_output=[]
    for data in structured_data:
        for row, row_text in data.items():
            final_output.extend([row_text])
    
    with open('/content/drive/MyDrive/YoloTrOCR/Table_extraction/images/output1.csv','w') as result_file:
        wr = csv.writer(result_file, dialect='excel')
    # The for loop MUST be inside the with statemen
        for row_text in final_output:
              wr.writerow(row_text)
    return final_output

# main("/content/drive/MyDrive/YoloTrOCR/Table_extraction/images/anmol.jpg")

