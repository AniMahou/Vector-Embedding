import unicodedata
import re
from typing import Optional

class EmbeddingPreprocessor:
    """Prepare text for embedding models."""
    
    def __init__(self, max_tokens: int = 8192):
        self.max_tokens = max_tokens
    
    def normalize_unicode(self, text: str) -> str:
        """Convert to NFC normalized form."""
        return unicodedata.normalize('NFC', text)
    
    def fix_encoding_artifacts(self, text: str) -> str:
        """Fix common web scraping artifacts."""
        replacements = {
            'â€™': "'",
            'â€œ': '"',
            'â€': '"',
            'â€¦': '...',
            'â€"': '-',
            'Â': '',
        }
        for bad, good in replacements.items():
            text = text.replace(bad, good)
        return text
    
    def clean_whitespace(self, text: str) -> str:
        """Normalize whitespace without removing line breaks."""
        # Replace multiple spaces with single space
        text = re.sub(r'[ \t]+', ' ', text)
        # Keep at most one blank line
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        # Trim outer whitespace
        return text.strip()
    
    def truncate_to_tokens(self, text: str, tokenizer) -> str:
        """Truncate text to max_tokens."""
        tokens = tokenizer.encode(text)
        if len(tokens) > self.max_tokens:
            tokens = tokens[:self.max_tokens]
            text = tokenizer.decode(tokens)
        return text
    
    def add_query_context(self, query: str, is_short: bool = True) -> str:
        """Add context to ambiguous short queries."""
        ambiguous_terms = {
            'apple': 'Apple company or apple fruit',
            'java': 'Java programming language or Java island',
            'python': 'Python programming language',
            'rust': 'Rust programming language',
            'go': 'Go programming language',
        }
        
        query_lower = query.lower().strip()
        if query_lower in ambiguous_terms:
            # Let the embedding model figure it out, but add a hint
            return f"{query} ({ambiguous_terms[query_lower]})"
        return query
    
    def process(self, text: str, is_query: bool = False) -> str:
        """Full preprocessing pipeline."""
        text = self.normalize_unicode(text)
        text = self.fix_encoding_artifacts(text)
        text = self.clean_whitespace(text)
        
        if is_query:
            text = self.add_query_context(text)
        
        return text

# Usage
preprocessor = EmbeddingPreprocessor()

# Raw scraped text
raw_text = "The product is â€œamazingâ€...Highly recommend!!!\n\n\n\nBuy it now!"

clean_text = preprocessor.process(raw_text)
print(f"Original: {raw_text}")
print(f"Cleaned:  {clean_text}")
# Output: "The product is "amazing"... Highly recommend!!!\n\nBuy it now!"