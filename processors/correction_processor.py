import nltk
import re
from typing import Union, List, Set
import google.generativeai as genai



class TextValidityChecker:
    """
    A class to check the validity of words in text based on a dictionary lookup.
    Uses NLTK's English word corpus by default but can accept custom dictionaries.
    """
    
    def __init__(self, threshold: float = 0.65, custom_dictionary: Set[str] = None):
        """
        Initialize the TextValidityChecker with a validity threshold.
        
        Args:
            threshold: The minimum ratio of valid words to consider text valid
            custom_dictionary: Optional custom dictionary to use instead of NLTK's
        """
        self.threshold = threshold
        self.dictionary = custom_dictionary if custom_dictionary is not None else self._load_nltk_dictionary()
        
    def _load_nltk_dictionary(self) -> Set[str]:
        """
        Load the NLTK English word dictionary.
        
        Returns:
            Set[str]: A set of lowercase English words
        """
        try:
            nltk.data.find('corpora/words')  # Check if words corpus is already downloaded
        except LookupError:
            print("Downloading NLTK 'words' corpus...")
            nltk.download('words')  # Download if not present
            
        from nltk.corpus import words
        return set(word.lower() for word in words.words())  # Create set, lowercase
    
    def is_valid_word(self, word: str) -> bool:
        """
        Check if a word is in the dictionary.
        
        Args:
            word: The word to check
            
        Returns:
            bool: True if the word is in the dictionary, False otherwise
        """
        processed_word = word.lower()  # Convert to lowercase for case-insensitivity
        return processed_word in self.dictionary
    
    def extract_words(self, text: str) -> List[str]:
        """
        Extract words from text using regex.
        
        Args:
            text: The text to extract words from
            
        Returns:
            List[str]: A list of extracted words
        """
        return re.findall(r"\b[\w]+\b", text)
    
    def check_text_validity(self, text_input: Union[str, List[str]], verbose: bool = True) -> bool:
        """
        Check if the given text contains a sufficient proportion of valid words.
        
        Args:
            text_input: Text to check, either as a string or a list of strings
            verbose: Whether to print the validity ratio
            
        Returns:
            bool: True if the text is valid according to the threshold, False otherwise
        """
        # Convert list to string if needed
        if isinstance(text_input, list):
            text = " ".join(text_input)
        else:
            text = text_input
            
        # Extract words
        extracted_words = self.extract_words(text)
        
        # Handle empty text
        if not extracted_words:
            if verbose:
                print("No words found in the text.")
            return False
            
        # Count valid words
        valid_count = sum(1 for word in extracted_words if self.is_valid_word(word))
        validity_ratio = valid_count / len(extracted_words)
        
        if verbose:
            print(f"\nValidity ratio: {validity_ratio:.2f} ({valid_count}/{len(extracted_words)} words)")
            
        # Check against threshold
        return validity_ratio >= self.threshold
    


    def api(self,image):

      api_error_message_blurry = "That's a very blurry and small image.  I cannot reliably extract any handwritten words from it.  The resolution is too low and the quality is too poor for accurate text recognition."

      print('/n Using api')
      try:
        genai.configure(api_key="your_api_key")

        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt_text = "Extract the handwritten words from this image.Donot output text other than that in image" 
        response = model.generate_content([prompt_text, image])

        if response and hasattr(response, 'text') and ( "cannot extract response" in response.text.lower() or api_error_message_blurry.lower() in response.text.lower()):
          return None 
        else:
          return response.text
        
      except Exception as e:
        print(f"Error calling Gemini API: {e}") 
        return None 
