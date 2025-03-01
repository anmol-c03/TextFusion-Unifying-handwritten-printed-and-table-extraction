#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for file operations.
"""

import os
import shutil

def ensure_directories(directory_paths):
    """
    Ensure that all specified directories exist.
    
    Args:
        directory_paths (list): List of directory paths to ensure
    """
    for path in directory_paths:
        os.makedirs(path, exist_ok=True)
    print('All directories exist')

def clean_directories(directory_paths):
    """
    Delete all files and subdirectories in the specified directories.
    
    Args:
        directory_paths (list): List of directory paths to clean
    """
    for folder_path in directory_paths:
        if os.path.exists(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                try:
                    if os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                    else:
                        os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
            print(f"Cleaned directory: {folder_path}")
        else:
            print(f"Directory does not exist: {folder_path}")


def sort_files_naturally(files):
    """
    Sort filenames that include numbers in natural order.
    
    Args:
        files (list): List of filenames to sort
        
    Returns:
        list: Sorted list of filenames
    """
    return sorted(files, key=lambda x: int(x.split('_')[1][:-4]))

    # import re
    
    # def atoi(text):
    #     return int(text) if text.isdigit() else text
    
    # def natural_keys(text):
    #     return [atoi(c) for c in re.split(r'(\d+)', text)]
    
    # return sorted(files, key=natural_keys)