import re
from typing import List, Dict, Optional

class TextQualityFilter:
    """Text quality filtering and preprocessing.
    
    Args:
        min_length (int): Minimum number of characters to be included as a sample.
        max_length (int): Maximum number of characters to be included as a sample.
    """
    def __init__(
        self,
        min_length: int = 100, 
        max_length: int = 8192
    ):
        self.min_length = min_length
        self.max_length = max_length

        # Regex patterns for filtering
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
        self.phone_pattern = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
        self.excessive_whitespace = re.compile(r'\s{3,}')
        self.excessive_newlines = re.compile(r'\n{4,}')
        self.excessive_punctuation = re.compile(r'[.!?]{4,}')
        self.non_printable = re.compile(r'[^\x20-\x7E\n\t]')

        # Language detection patterns
        self.english_ratio_threshold = 0.7

    def is_english(self, text: str) -> bool:
        """Simple English detection based on character distribution.
        
        Args:
            text (str): String containing input text.

        Returns:
            bool: Whether the text is English or not.
        """
        # Empty strings, not English
        if not text:
            return False

        # Count ASCII and alphabetical values
        ascii_letters = sum(1 for c in text if c.isalpha() and ord(c) < 128)
        total_chars = len([c for c in text if c.isalpha()])

        # Return False if no alphabetical characters
        if total_chars == 0:
            return False

        # Check ratio of ASCII values to total values
        return (ascii_letters / total_chars) >= self.english_ratio_threshold

    def clean_text(self, text: str) -> str:
        """Clean and normalize text.
        
        Args:
            text (str): String containing input text.

        Returns:
            str: String of cleaned/normalized text.
        """
        text = self.url_pattern.sub(' ', text)
        text = self.email_pattern.sub(' ', text)
        text = self.phone_pattern.sub(' ', text)
        text = self.non_printable.sub(' ', text)
        text = self.excessive_whitespace.sub(' ', text)
        text = self.excessive_newlines.sub('\n\n', text)
        text = self.excessive_punctuation.sub('...', text)
        text = re.sub(r' +\n', '\n', text)
        text = re.sub(r'\n +', '\n', text)
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"['']", "'", text)
        return text.strip()

    def calculate_quality_score(self, text: str) -> float:
        """Calculate text quality score .
        
        Args:
            text (str): String containing input text.

        Returns:
            float: Calulated quality score in the range [0, 1], closer to 1 is better.
        """
        # No text, return score of 0
        if not text:
            return 0.0

        # Initialize score at 1
        score = 1.0
        length = len(text)
        if length < self.min_length:
            score = score * (length / self.min_length)
        elif length > self.max_length:
            score = score * (self.max_length / length)

        # Ensure there are at least 10 words
        words = text.split()
        if len(words) < 10:
            score = 0.5 * score

        # Ensure average word length between 3-12
        avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
        if avg_word_length < 3 or avg_word_length > 12:
            score *= 0.8

        # Ensure there are at least 3 sentences
        sentences = re.findall(r'[^.!?]+[.!?]', text)
        if len(sentences) < 3:
            score *= 0.7

        # Check each new line
        lines = text.split('\n')
        unique_lines = set(line.strip() for line in lines if line.strip())
        if len(lines) > 0 and len(unique_lines) / len(lines) < 0.8:
            score *= 0.6

        # Check if punctuation ratio is reasonable
        punct_count = sum(1 for c in text if c in '.,!?;:')
        punct_ratio = punct_count / max(len(text), 1)
        if punct_ratio < 0.01 or punct_ratio > 0.1:
            score *= 0.8

        # Ensure not too many capital letters
        caps_count = sum(1 for c in text if c.isupper())
        caps_ratio = caps_count / max(len(text), 1)
        if caps_ratio > 0.2:
            score *= 0.7

        return score

    def filter_text(self, text: str, min_quality: float = 0.5) -> Optional[str]:
        """Filter and clean text based on quality criteria.
        
        Args:
            text (str): String of input text.
            min_quality (float): Minimum quality score from 0-1 for the text to be kept.
        
        Returns:
            Optional[str]: Returns None if any test fails, or returns cleaned text if all passes.
        """
        if not text or len(text) < self.min_length:
            return None

        if not self.is_english(text):
            return None

        cleaned = self.clean_text(text)
        if not cleaned or len(cleaned) < self.min_length:
            return None

        quality = self.calculate_quality_score(cleaned)
        if quality < min_quality:
            return None

        return cleaned

    def __call__(
        self, 
        batch: Dict[str, List[str]], 
        min_quality: float = 0.5
    ) -> Dict[str, List[Optional[str]]]:
        """Support for datasets.map with batching.
        
        Args:
            batch (Dict[str, List[str]]): Batch of texts samples and corresponding sample numbers.
            min_quality (float): Minimum quality score from 0-1 for the text to be kept.

        Returns:
            Dict[str, List[Optional[str]]]: Filtered text where bad data is None.
        """
        return {
            "text": [
                self.filter_text(text, min_quality=min_quality)
                for text in batch["text"]
            ]
        }
    