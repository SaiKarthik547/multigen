"""
SemanticCompressor — Phase 13 Target

Performs heuristic semantic extraction to compress long narrative prompts
into dense, token-efficient diffusion prompts without losing key details.
"""

import re
from typing import List
from multigenai.core.logging.logger import get_logger

LOG = get_logger(__name__)

class SemanticCompressor:
    """
    Compresses long prompts into dense semantic descriptors.
    Removes filler words, extracts noun phrases, and respects token budgets.
    """

    _STOPWORDS = frozenset({
        "a", "an", "the", "is", "are", "was", "were", "in", "on", "at", "by", 
        "for", "with", "of", "and", "or", "but", "to", "through", "while", 
        "during", "approaches", "enters", "walks", "runs", "goes", "moves",
        "filled", "that", "this", "these", "those", "it", "they", "he", "she",
        "we", "you", "has", "have", "had", "been", "being", "am", "do", "does", "did"
    })

    def __init__(self, target_tokens: int = 60):
        self.target_tokens = target_tokens

    def compress(self, prompt: str) -> str:
        """
        Extract key phrases and return a dense, comma-separated string.
        """
        # 1. Clean up spacing and newlines
        text = re.sub(r'\s+', ' ', prompt).strip()
        
        # 2. Split into chunks based on punctuation
        raw_chunks = re.split(r'[,.!?;]+', text)
        
        extracted_phrases = []
        
        for chunk in raw_chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
                
            words = chunk.split()
            current_phrase = []
            
            for word in words:
                clean_word = re.sub(r'[^\w\-]', '', word).lower()
                
                if clean_word in self._STOPWORDS:
                    if current_phrase:
                        extracted_phrases.append(" ".join(current_phrase))
                        current_phrase = []
                else:
                    # Strip punctuation from the original word but keep its capitalization
                    current_phrase.append(word.strip(',.!?;:()[]{}'))
            
            if current_phrase:
                extracted_phrases.append(" ".join(current_phrase))
                
        # 3. Filter out single-character artifacts and deduplicate (case-insensitive)
        seen = set()
        final_phrases = []
        for phrase in extracted_phrases:
            if len(phrase) < 2:
                continue
                
            phrase_lower = phrase.lower()
            if phrase_lower not in seen:
                seen.add(phrase_lower)
                final_phrases.append(phrase)
                
        # 4. Rebuild prompt respecting the budget
        compressed_text = ""
        current_tokens = 0
        
        for phrase in final_phrases:
            phrase_tokens = len(phrase.split())
            if current_tokens + phrase_tokens > self.target_tokens:
                break
                
            if compressed_text:
                compressed_text += ", " + phrase
            else:
                compressed_text = phrase
                
            current_tokens += phrase_tokens
            
        LOG.info(f"SemanticCompressor: reduced from {len(prompt.split())} to {current_tokens} words.")
        return compressed_text
