import re
import nltk
from typing import List, Dict, Any, Tuple
from collections import Counter
import logging

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except:
        pass

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    try:
        nltk.download('stopwords', quiet=True)
    except:
        pass

def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    """Extract key terms from text for better chunking decisions."""
    try:
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        
        # Get English stopwords
        stop_words = set(stopwords.words('english'))
        
        # Tokenize and filter
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalpha() and len(word) > 2 and word not in stop_words]
        
        # Count frequency
        word_freq = Counter(words)
        
        # Return top keywords
        return [word for word, freq in word_freq.most_common(top_k)]
        
    except Exception:
        # Fallback simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        filtered_words = [word for word in words if word not in common_words]
        return list(Counter(filtered_words).keys())[:top_k]

def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """Calculate semantic similarity between two texts using keyword overlap."""
    try:
        keywords1 = set(extract_keywords(text1, 20))
        keywords2 = set(extract_keywords(text2, 20))
        
        if not keywords1 or not keywords2:
            return 0.0
        
        # Jaccard similarity
        intersection = keywords1.intersection(keywords2)
        union = keywords1.union(keywords2)
        
        return len(intersection) / len(union) if union else 0.0
        
    except Exception:
        return 0.0

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using NLTK or fallback regex."""
    try:
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)
    except Exception:
        # Fallback regex-based sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

def split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs."""
    paragraphs = re.split(r'\n\s*\n', text.strip())
    return [p.strip() for p in paragraphs if p.strip()]

def calculate_text_complexity(text: str) -> Dict[str, float]:
    """Calculate various text complexity metrics."""
    try:
        sentences = split_into_sentences(text)
        words = re.findall(r'\b\w+\b', text)
        
        metrics = {
            'sentence_count': len(sentences),
            'word_count': len(words),
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'unique_word_ratio': len(set(words)) / len(words) if words else 0
        }
        
        return metrics
        
    except Exception:
        return {
            'sentence_count': 0,
            'word_count': 0,
            'avg_sentence_length': 0,
            'avg_word_length': 0,
            'unique_word_ratio': 0
        }

def identify_section_boundaries(text: str) -> List[int]:
    """Identify natural section boundaries in text."""
    boundaries = []
    
    # Split into paragraphs
    paragraphs = split_into_paragraphs(text)
    current_pos = 0
    
    for i, paragraph in enumerate(paragraphs):
        # Check for section indicators
        if re.match(r'^\s*\d+\.\s+', paragraph):  # Numbered sections
            boundaries.append(current_pos)
        elif re.match(r'^\s*[A-Z][A-Z\s]+:?\s*$', paragraph):  # ALL CAPS headers
            boundaries.append(current_pos)
        elif re.match(r'^\s*#{1,6}\s+', paragraph):  # Markdown headers
            boundaries.append(current_pos)
        elif paragraph.endswith(':') and len(paragraph) < 100:  # Short lines ending with colon
            boundaries.append(current_pos)
        
        current_pos += len(paragraph) + 2  # +2 for paragraph separator
    
    return boundaries

def smart_text_split(text: str, target_size: int, max_size: int, overlap: int = 200) -> List[str]:
    """Smart text splitting that respects sentence and paragraph boundaries."""
    if len(text) <= target_size:
        return [text]
    
    chunks = []
    
    # First, try to split by section boundaries
    section_boundaries = identify_section_boundaries(text)
    
    if section_boundaries:
        # Split by sections first
        sections = []
        start = 0
        for boundary in section_boundaries + [len(text)]:
            if boundary > start:
                sections.append(text[start:boundary])
                start = boundary
        
        # Process each section
        for section in sections:
            if len(section) <= max_size:
                chunks.append(section)
            else:
                # Further split large sections
                sub_chunks = _split_by_paragraphs(section, target_size, max_size, overlap)
                chunks.extend(sub_chunks)
    else:
        # No clear sections, split by paragraphs
        chunks = _split_by_paragraphs(text, target_size, max_size, overlap)
    
    # Add overlap between chunks
    if len(chunks) > 1:
        chunks = _add_overlap(chunks, overlap)
    
    return chunks

def _split_by_paragraphs(text: str, target_size: int, max_size: int, overlap: int) -> List[str]:
    """Split text by paragraphs, respecting size constraints."""
    paragraphs = split_into_paragraphs(text)
    chunks = []
    current_chunk = []
    current_size = 0
    
    for paragraph in paragraphs:
        para_size = len(paragraph)
        
        # If adding this paragraph exceeds max size, finalize current chunk
        if current_size + para_size > max_size and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            # Start new chunk with some overlap
            if overlap > 0 and current_chunk:
                overlap_text = current_chunk[-1][-overlap:] if len(current_chunk[-1]) > overlap else current_chunk[-1]
                current_chunk = [overlap_text]
                current_size = len(overlap_text)
            else:
                current_chunk = []
                current_size = 0
        
        # If single paragraph is too large, split by sentences
        if para_size > max_size:
            sentence_chunks = _split_by_sentences(paragraph, target_size, max_size)
            chunks.extend(sentence_chunks)
        else:
            current_chunk.append(paragraph)
            current_size += para_size + 2  # +2 for paragraph separator
    
    # Add remaining content
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks

def _split_by_sentences(text: str, target_size: int, max_size: int) -> List[str]:
    """Split text by sentences when paragraphs are too large."""
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence)
        
        if current_size + sentence_size > max_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0
        
        current_chunk.append(sentence)
        current_size += sentence_size + 1  # +1 for space
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def _add_overlap(chunks: List[str], overlap: int) -> List[str]:
    """Add overlap between consecutive chunks."""
    if overlap <= 0 or len(chunks) < 2:
        return chunks
    
    overlapped_chunks = [chunks[0]]
    
    for i in range(1, len(chunks)):
        prev_chunk = chunks[i-1]
        current_chunk = chunks[i]
        
        # Get overlap from previous chunk
        if len(prev_chunk) > overlap:
            overlap_text = prev_chunk[-overlap:]
            overlapped_chunk = overlap_text + '\n\n' + current_chunk
        else:
            overlapped_chunk = current_chunk
        
        overlapped_chunks.append(overlapped_chunk)
    
    return overlapped_chunks

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.,!?;:()\[\]"-]', '', text)
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    return text.strip()