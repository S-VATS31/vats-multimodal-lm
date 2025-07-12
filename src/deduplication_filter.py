import re
import hashlib
from typing import List, Dict, Optional

class DeduplicationFilter:
    """Efficient deduplication using hashing.
    
    Args:
        similarity_threshold (float): Cutoff used to check the similarity between 2 samples.
    """
    def __init__(self, similarity_threshold: float = 0.85):
        self.seen_hashes = set()
        self.similarity_threshold = similarity_threshold

    def get_text_hash(self, text: str) -> str:
        """Generate hash for text.
        
        Args:
            text (str): String of input text.
        
        Returns:
            str: Returns hashed text.
        """
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()

    def get_shingles(self, text: str, k: int = 5) -> set:
        """Generate k-shingles for similarity detection.
        
        Args:
            text: String of input text.
            k: Consecutive number of tokens to be grouped to detect similarity.
        
        Returns:
            set: Set of consecutive pairs to check for duplicates.
        """
        words = text.lower().split()
        if len(words) < k:
            return {' '.join(words)}
        return {' '.join(words[i:i+k]) for i in range(len(words) - k + 1)}

    def is_duplicate(self, text: str) -> bool:
        """Check if text is a duplicate or near-duplicate.
        
        Args:
            text (str): String of input text.
        
        Returns:
            bool: True if duplicate, else False.
        """
        text_hash = self.get_text_hash(text)

        if text_hash in self.seen_hashes:
            return True

        self.seen_hashes.add(text_hash)
        return False

    def __call__(
        self,
        batch: Dict[str, List[Optional[str]]]
    ) -> Dict[str, List[Optional[str]]]:
        """Support for datasets.map with batching.
        
        Args:
            batch (Dict[str, List[Optional[str]]]): Input batch of texts (None means invalid).
        
        Returns:
            Dict[str, List[Optional[str]]]: Returns output batch with deduplication applied.
        """
        deduped = []
        for text in batch["text"]:
            if text is not None and not self.is_duplicate(text):
                deduped.append(text)
            else:
                deduped.append(None)
        return {"text": deduped}