#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Text processing module for OCR using PaddleOCR and TrOCR.
"""

import os
import cv2
import re
import numpy as np
import tiktoken
from PIL import Image
from paddleocr import PaddleOCR
from utils.file_utils import sort_files_naturally
from processors.text_recognition import TextRecognition
from processors.text_detection import TextDetection



class TextProcessor:
    """Class to handle text processing operations."""
    
    def __init__(self, confidence_threshold=0.9):
        """
        Initialize the text processor.
        
        Args:
            confidence_threshold (float): Threshold for PaddleOCR confidence
        """
        self.confidence_threshold = confidence_threshold
        
        # Initialize OCR engines
        self.paddle_ocr = PaddleOCR(
            det_db_thresh=0.3,
            det_db_box_thresh=0.5,
            det_db_unclip_ratio=1.6,
            use_dilation=True,
            use_angle_cls=True,
            lang='en'
        )
        
        # Initialize TrOCR for handwritten text
        self.tr_ocr = TextRecognition()
        
        # Token mapping for better recognition
        self.token_mapping = {
            'pulchowk': '!',
            'tribhuvan': '@',
            'msdsa': '1'
        }
        
        # Initialize tokenizer for checking handwritten vs printed
        self.tokenizer = tiktoken.get_encoding('cl100k_base')
    
    def process_directory(self, directory_path):
        """
        Process all images in a directory.
        
        Args:
            directory_path (str): Path to directory containing images
            
        Returns:
            list: List of dictionaries with processing results
        """
        print(f"Processing text in images from: {directory_path}")
        
        # Get all image files and sort them
        image_files = [f for f in os.listdir(directory_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        image_files = sort_files_naturally(image_files)
        print('sorted_image_files_are:',image_files)
        
        results = []
        for idx, img_file in enumerate(image_files):
            image_path = os.path.join(directory_path, img_file)
            result = self.process_image(idx, image_path)
            results.append(result)
            print(f"  Processed {img_file}: {'Handwritten' if result['is_handwritten'] else 'Printed'}")

        for result in results:
          print(result['filtered_results'],result['text'],result['is_handwritten'],result['image_path'])     
        corrected_results=self.process_handwritten_texts(results)

        for result in corrected_results:
          print(result['text'])
        return corrected_results
    
    def process_image(self, idx, img_path):
        """
        Process a single image.
        
        Args:
            idx (int): Index of the image
            image_path (str): Path to the image
            
        Returns:
            dict: Dictionary with processing results
        """
        # Run PaddleOCR
        is_handwritten, filtered_results, extracted_texts = self.recognize_text(img_path)
        

        # Process text based on handwritten flag
        return {
            'image_path': img_path,
            'is_handwritten': is_handwritten,
            'filtered_results': filtered_results,
            'text': extracted_texts
        }
    def recognize_text(self, image_path):
        """
        Recognize text using PaddleOCR and determine if it's handwritten.
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            tuple: (is_handwritten, filtered_results, extracted_texts)
        """
        # Run OCR
        result = self.paddle_ocr.ocr(image_path, cls=True)
        
        # Check if OCR found anything
        if result is None or not result or not result[0]:
            return 1, [], []  # No text detected, assume handwritten
        
        # Sort OCR results in reading order
        result = self.sort_ocr_results(result[0])
        
        # Extract text and filter low confidence results
        filtered_results = []
        extracted_texts = []
        
        for line in result:
            bbox, (text, score) = line
            print(score)
            extracted_texts.append(text)
            # FIX: Add to filtered_results when score is BELOW the threshold
            if score < 0.9:
                filtered_results.append((bbox, text, score))  # Note: Changed order to match usage in process_handwritten_texts

        # Check if text appears to be handwritten
        
        is_handwritten = self.check_if_handwritten(extracted_texts)

        return (is_handwritten,filtered_results, extracted_texts)

    
    def sort_ocr_results(self, results):
        """
        Sort OCR results in reading order (top-to-bottom, left-to-right).
        
        Args:
            results: OCR results from PaddleOCR
            
        Returns:
            list: Sorted OCR results
        """
        boxes_and_texts = []
        for line in results:
            box = line[0]
            text = line[1][0]
            confidence = line[1][1]

            # Calculate center y-coordinate (for vertical sorting)
            center_y = sum(point[1] for point in box) / 4
            # Calculate leftmost x-coordinate (for horizontal sorting)
            left_x = min(point[0] for point in box)

            boxes_and_texts.append({
                'box': box,
                'text': text,
                'confidence': confidence,
                'center_y': center_y,
                'left_x': left_x
            })

        # Group text by rows (texts with similar y values)
        y_threshold = 20  # Adjust based on text size and spacing

        # Sort by y-coordinate first
        boxes_and_texts.sort(key=lambda x: x['center_y'])

        # Group into rows
        rows = []
        if boxes_and_texts:
            current_row = [boxes_and_texts[0]]

            for item in boxes_and_texts[1:]:
                if abs(item['center_y'] - current_row[0]['center_y']) < y_threshold:
                    # Same row
                    current_row.append(item)
                else:
                    # New row
                    rows.append(sorted(current_row, key=lambda x: x['left_x']))
                    current_row = [item]

            # Don't forget the last row
            if current_row:
                rows.append(sorted(current_row, key=lambda x: x['left_x']))

        # Flatten the rows into a single sorted list
        sorted_results = []
        for row in rows:
            for item in row:
                # Recreate the original format
                sorted_results.append([item['box'], [item['text'], item['confidence']]])

        return sorted_results
    
    def check_if_handwritten(self, text_list):
        """
        Check if text appears to be handwritten based on tokenization.
        
        Args:
            text_list (list): List of extracted text strings
            
        Returns:
            int: 1 if handwritten, 0 if printed
        """
        if not text_list:
            return 1  # No text detected, assume handwritten
        
        # Join all text lines
        text = '\n'.join(text_list)
        
        # Replace known tokens
        text = self.replace_tokens(text.lower())
        
        # Extract words
        words = re.findall(r'[a-zA-Z]+', text)
        if not words:
            return 0  # No words found, likely not handwritten
        
        # Tokenize
        tokens = self.tokenizer.encode(' '.join(words))
        if not tokens:
            return 0
        
        # Calculate tokenization granularity
        tokenization_granularity = len(words) / len(tokens)
        
        # Heuristic: handwritten text typically has higher tokenization granularity
        if tokenization_granularity >= 0.65 or len(words) == 1:
            return 0  # Likely printed text
        
        return 1  # Likely handwritten
    
    def replace_tokens(self, text):
        """
        Replace known tokens in text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with tokens replaced
        """
        for token, token_id in self.token_mapping.items():
            text = text.replace(token, token_id)
        return text
    

    
