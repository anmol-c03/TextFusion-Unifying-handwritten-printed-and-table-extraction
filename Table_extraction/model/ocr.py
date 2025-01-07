import numpy as np
from paddleocr import PaddleOCR
from tqdm.auto import tqdm
from PIL import Image
from cell_cordinates import cell_coordinates
from crop_table import cropped_table

import csv
import numpy as np
from paddleocr import PaddleOCR
from tqdm.auto import tqdm
from PIL import Image
# from cell_cordinates import cell_coordinates
# from crop_table import cropped_table

import csv

import logging
logging.getLogger().setLevel(logging.CRITICAL)

