#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script to run the OCR pipeline.
"""

import os
import argparse
from utils.file_utils import ensure_directories, clean_directories
from processors.pdf_processor import PDFProcessor
from processors.layout_processor import LayoutProcessor
from processors.text_processor import TextProcessor
from processors.correction_processor import CorrectionProcessor

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Document OCR Pipeline')
    parser.add_argument('--input', type=str, required=True, help='Path to input PDF or image file')
    parser.add_argument('--output', type=str, default='txt', help='Path to output text folder')
    parser.add_argument('--model_path', type=str, default='./model_doclayout/DocLayout-YOLO-DocStructBench/doclayout_yolo_docstructbench_imgsz1024.pt',
                        help='Path to DocLayout YOLO model')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for PDF conversion')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold for text detection')
    parser.add_argument('--clean', action='store_true', help='Clean temp directories before processing')
    parser.add_argument('--visualize', action='store_true', help='Visualize detected regions')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Setup directories
    root_path = os.path.abspath(os.getcwd())
    dirs = {
        'original': os.path.join(root_path, 'images', 'original'),
        'resized': os.path.join(root_path, 'images', 'resized'),
        'enhanced': os.path.join(root_path, 'images', 'enhanced'),
        'visualization': os.path.join(root_path, 'images', 'visualization'),
        'extract_pdf': os.path.join(root_path, 'images', 'extract_pdf'),
        'text': os.path.join(root_path, 'txt')
    }
    
    ensure_directories(list(dirs.values()))
    
    if args.clean:
        clean_directories([dirs['original'], dirs['resized'], dirs['enhanced'], dirs['visualization']])
    
    # Process the document
    input_path = args.input
    
    # Step 1: Convert PDF to images if needed
    if input_path.lower().endswith('.pdf'):
        pdf_processor = PDFProcessor(dpi=args.dpi)
        pdf_processor.convert_to_images(input_path, dirs['extract_pdf'])
        # Update input path to the first extracted image
        input_files = sorted(os.listdir(dirs['extract_pdf']))
    '''
    TODO:
    add code to process single images,too.
    '''
    # elif input_path.lower().endswith('.jpg'):
    #     input_files = [input_path]

    output_folder = args.output if os.path.isdir(args.output) else dirs['text']
    ensure_directories([output_folder])

    for path in input_files:
        try:
            pdf2img_path=os.path.join(dirs['extract_pdf'], path)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue
        # Step 2: Layout analysis
        if args.clean:
            clean_directories([dirs['original'], dirs['resized'], dirs['enhanced'], dirs['visualization']])
    
        layout_processor = LayoutProcessor(model_path=args.model_path, img_path=pdf2img_path)
        layout_processor.crop_images()
        layout_processor.visualize_bbox()
        
        # Step 3: OCR processing
        text_processor = TextProcessor(confidence_threshold=args.confidence)
        final_texts = text_processor.process_directory(dirs['original'])
        
        # for image in image_results:
        #     print(image['text'])
        # Step 4: Text correction
        # correction_processor = CorrectionProcessor()
        # final_texts = correction_processor.correct_all(image_results)
        
        # Save results
        txt_filename = os.path.join(output_folder, f"{os.path.basename(pdf2img_path).split('.')[0]}.txt")
        with open(txt_filename, 'w', encoding='utf-8') as file:
            for text in final_texts:
                file.write(f"{text}\n")
    
    print(f"Processing complete! Results saved to {args.output}")

if __name__ == "__main__":
    main()