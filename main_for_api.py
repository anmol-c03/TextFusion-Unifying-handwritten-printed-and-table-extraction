#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script to run the OCR pipeline.
"""

import os
import argparse
import pandas as pd
from utils.file_utils import ensure_directories, clean_directories
from processors.pdf_processor import PDFProcessor
from processors.layout_processor import LayoutProcessor
from processors.text_processor import TextProcessor
from processors.correction_processor import TextValidityChecker
from Table_extraction.main import extract


model_path="./model_doclayout/DocLayout-YOLO-DocStructBench/doclayout_yolo_docstructbench_imgsz1024.pt"

def main(img_path="survey_forms/report.JPG"):
    # args = parse_arguments()
    
    # Setup directories
    root_path = os.path.abspath(os.getcwd())
    dirs = {
        'original': os.path.join(root_path, 'images', 'original'),
        'resized': os.path.join(root_path, 'images', 'resized'),
        'visualization': os.path.join(root_path, 'images', 'visualization'),
        'text': os.path.join(root_path, 'txt')
    }
    
    ensure_directories(list(dirs.values()))
    
    clean_directories([dirs['original'], dirs['resized'], dirs['visualization']])
    
    # Process the document
    input_path = img_path

    output_folder =os.makedirs('txt',exist_ok=True)
    # ensure_directories([output_folder])

        # Step 2: Layout analysis

    clean_directories([dirs['original'], dirs['resized'], dirs['visualization']])

    layout_processor = LayoutProcessor(model_path=model_path, img_path=input_path)
    layout_processor.crop_images()
    layout_processor.visualize_bbox()
    
    # Step 3: OCR processing
    text_processor = TextProcessor()
    image_results,ocr = text_processor.process_directory(dirs['original'])

    for image in image_results:
      if 'Table' in os.path.basename(image['image_path']):
        outputs=extract(ocr,image['image_path'])
        df = pd.DataFrame(outputs[1:], columns=outputs[0])
        image['text']=df.to_string(index=False)



    
    # Step 4: Text correction
    # correction_processor = TextValidityChecker()
    # final_texts = correction_processor.correct_all(image_results)
    
    # Save results
    txt_filename = "/content/drive/MyDrive/Major_project/YoloTrOCR/txt/page_4.txt"
    with open(txt_filename, 'w', encoding='utf-8') as file:
        for text in image_results:
            file.write(f"{text['text']}\n")
    
    for image in image_results:
      print(image['text'])

if __name__ == "__main__":
    main()