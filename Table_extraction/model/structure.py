import torch
from transformers import TableTransformerForObjectDetection
from torchvision import transforms
from load import device
from preprocess import MaxResize,outputs_to_objects
from crop_table import cropped_table

# new v1.1 checkpoints require no timm anymore
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






