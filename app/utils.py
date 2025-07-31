"""
Utility functions for the Streamlit app
"""

import re
import string

def clean_arabic_text_minimal(text):
    """Basic Arabic text cleaning for the Streamlit app"""
    if not text:
        return ""
    
    # Basic cleaning
    text = str(text).strip()
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove some punctuation (keep Arabic punctuation)
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
    
    # Remove extra spaces again
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
