import re
from app.core.constants import TECH_TERM_MAP, NOISE_PATTERNS

def clean_system_noise(text: str) -> str:
    """
    Removes system brackets, music metadata tags, and sanitizes spacing.
    """
    if not text:
        return ""
    
    text = text.replace('\n', ' ')
    
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def normalize_tech_terms(text: str) -> str:
    """
    Maps phonetic AI misspellings directly to standard enterprise tech terms.
    Uses word-boundary checks to prevent corrupting regular words.
    """
    if not text:
        return ""
        
    normalized_text = text
    for phonetic, correct_term in TECH_TERM_MAP.items():
        pattern = r'\b' + re.escape(phonetic) + r'\b'
        compiled_regex = re.compile(pattern, re.IGNORECASE)
        normalized_text = compiled_regex.sub(correct_term, normalized_text)
        
    return normalized_text

def preprocess_segment(raw_text: str) -> str:
    """
    Unified Pipeline Entrypoint for Preprocessing Layer.
    """
    clean_text = clean_system_noise(raw_text)
    final_text = normalize_tech_terms(clean_text)
    
    return final_text