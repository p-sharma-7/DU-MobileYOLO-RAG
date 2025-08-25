import numpy as np
import re

def simple_tokenize(text):
    """
    Tokenize text into lowercase words using regex.
    """
    return re.findall(r'\b\w+\b', text.lower())

def dummy_word_to_angle(word):
    """
    Map a word to a unique angle in radians between 0 and π.
    """
    return (hash(word) % 1000) / 1000 * np.pi

def text_to_quantum_angles(text, max_len=4):
    """
    Convert a text string into a fixed-length list of rotation angles.

    Parameters:
    - text: Input query or document string
    - max_len: Maximum number of tokens to encode

    Returns:
    - List of `max_len` angles (padded with 0.0 if shorter)
    """
    tokens = simple_tokenize(text)[:max_len]
    angles = [dummy_word_to_angle(tok) for tok in tokens]

    while len(angles) < max_len:
        angles.append(0.0)

    return angles
