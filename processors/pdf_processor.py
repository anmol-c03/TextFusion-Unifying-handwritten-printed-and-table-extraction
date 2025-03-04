#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PDF processing module for converting PDFs to images.
"""

import os
from pdf2image import convert_from_path

class PDFProcessor:
    """Class to handle PDF processing operations."""
    
    def __init__(self, dpi=300):
        """
        Initialize the PDF processor.
        
        Args:
            dpi (int): Dots per inch for image conversion
        """
        self.dpi = dpi
    
    def convert_to_images(self, pdf_path, output_folder):
        """
        Convert each page of a PDF into an image and save them.
        
        Args:
            pdf_path (str): Path to the input PDF file
            output_folder (str): Folder to save extracted images
            
        Returns:
            list: Paths to the saved images
        """
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"Converting PDF: {pdf_path}")
        images = convert_from_path(pdf_path, dpi=self.dpi)
        
        image_paths = []
        for i, img in enumerate(images, start=1):
            img_path = f"{output_folder}/page_{i}.png"
            img.save(img_path, "PNG")
            image_paths.append(img_path)
            print(f"  Saved: {img_path}")
            
        return image_paths