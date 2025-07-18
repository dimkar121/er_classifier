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

def check_if_match(row, gold_standard):
    """
    Checks if an ID pair from a row exists in the gold standard dictionary.
    Returns 1 if it's a match, 0 otherwise.
    """
    id_a = row['id_A']
    id_b = row['id_B']

    # Get the list of known matches for id_a, or an empty list if id_a is not in the gold standard
    matching_ids_for_a = gold_standard.get(id_a, [])
    if id_a not in gold_standard:
        return np.nan
    # Check if id_b is in that list of matches
    if id_b in matching_ids_for_a:
        return 1
    else:
        return 0


from datasketch import MinHash

def get_minhash(text, num_perm=128, k=3):
    """
    Generates a MinHash object for a given text.
    
    Args:
        text (str): The input string.
        num_perm (int): The number of permutation functions to use.
        k (int): The size of each shingle (character n-gram).
    """
    # Ensure input is a string
    text = str(text)
    
    # Create a MinHash object
    m = MinHash(num_perm=num_perm)
    
    # Create a set of shingles from the text
    shingles = {text[i:i+k] for i in range(len(text) - k + 1)}
    
    # Update the MinHash object with each shingle
    for s in shingles:
        m.update(s.encode('utf8'))
        
    return m

