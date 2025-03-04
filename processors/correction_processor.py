#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Text correction module using language tools.
"""

import language_tool_python

class CorrectionProcessor:
    """Class to handle text correction."""
    
    def __init__(self):
        """Initialize the text correction processor."""
        self.tool = language_tool_python.LanguageTool("en-US")
    
    def correct_all(self, image_results):
        """
        Correct all text in the image results.
        
        Args:
            image_results (list): List of dictionaries with image processing results
            
        Returns:
            list: List of corrected texts
        """
        corrected_texts = []
        for image_data in image_results:
            text = image_data['text']
            corrected_text, _ = self.check_and_correct_sentence(text)
            corrected_texts.append(corrected_text)
            
            # Update the original result dictionary
            image_data['text'] = corrected_text
        
        return corrected_texts
    
    def check_and_correct_sentence(self, sentence):
        """
        Check and correct grammatical errors in a sentence.
        
        Args:
            sentence (str): Input sentence to correct
            
        Returns:
            tuple: (corrected_sentence, corrections)
        """
        if not sentence:
            return sentence, []
            
        matches = self.tool.check(sentence)
        if not matches:
            return sentence, ["Correct sentence"]

        corrections = []
        for match in matches:
            incorrect_word = match.context[match.offset:match.offset + match.errorLength]
            suggestion = match.replacements[0] if match.replacements else "No suggestion"
            corrections.append(f"'{incorrect_word}' → '{suggestion}'")

        corrected_sentence = language_tool_python.utils.correct(sentence, matches)
        return corrected_sentence, corrections