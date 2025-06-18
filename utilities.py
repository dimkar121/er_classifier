import numpy as np
import mmh3
def bigrams(s):
    """Generates a set of 2-character shingles from a string."""
    if not isinstance(s, str):
        return set()
    s = s.lower()
    return {s[i:i + 2] for i in range(len(s) - 1)}


def minhash(set_elements, num_hashes=120):
    """
    Generates a binary MinHash signature for a set of elements.
    """
    # Handle empty sets
    if not set_elements:
        return np.zeros(num_hashes, dtype=np.uint8)

    signature = np.zeros(num_hashes, dtype=np.uint8)
    for i in range(num_hashes):
        # Calculate the minimum hash for the current hash function (seed)
        min_hash = min(mmh3.hash(str(el), seed=i) & 0xFFFFFFFF for el in set_elements)
        # Convert the result to a binary feature (1 if even, 0 if odd)
        signature[i] = 1 if (min_hash % 2) == 0 else 0
    return signature

