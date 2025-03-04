import numpy as np
from paddleocr import PaddleOCR
from tqdm.auto import tqdm
from PIL import Image



# from crop_table import cropped_table

import csv

import logging
logging.getLogger().setLevel(logging.CRITICAL)

# Initialize the PaddleOCR model (English)
class Recognize:
  def __init__(self,ocr):
    self.ocr=ocr

  def apply_ocr(self,cell_coordinate,crop):
      # Let's OCR row by row
      ocr = PaddleOCR(use_angle_cls=True, lang='en')  # You can add more languages if needed
      structured_data = []

      data = dict()
      max_num_columns = 0

      for idx, row in enumerate(tqdm(cell_coordinate)):
          row_text = []
          for cell in row["cells"]:
              cell_image = np.array(crop.crop(cell["cell"]))

              result = self.ocr.ocr(cell_image)


              if result ==[None] :
                  row_text.append("")
              else:
                  text = " ".join([line[1][0] for line in result[0]])
                  row_text.append(text)

                  

          if len(row_text) > max_num_columns:
              max_num_columns = len(row_text)

          data[idx] = row_text

      print("Max number of columns:", max_num_columns)

      for row, row_data in data.copy().items():
          if len(row_data) != max_num_columns:
              row_data = row_data + ["" for _ in range(max_num_columns - len(row_data))]
          data[row] = row_data

      return data



