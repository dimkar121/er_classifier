import pandas as pd
import numpy as np
import os
import re
from sentence_transformers import SentenceTransformer
import warnings
import mmh3  # Added for your custom MinHash function

# Suppress the specific warning from pandas about inplace renaming
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="`inplace` is deprecated and will be removed in a future version."
)


# --- Your Custom MinHash Implementation ---
# NOTE: This implementation requires the 'mmh3' library.
# You can install it by running: pip install mmh3

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


def get_minhash_vector(text):
    """
    A wrapper function that takes text, creates bigrams, generates a
    binary MinHash signature, and returns it as a list.
    """
    bigram_set = bigrams(text)
    signature_array = minhash(bigram_set)
    return signature_array.tolist()


def process_and_embed_table(input_filename, output_filename, model):
    """
    Loads a table, generates dense embeddings and MinHash vectors,
    and saves the result to a new file.
    """
    if not os.path.exists(input_filename):
        print(f"Error: Input file '{input_filename}' not found. Skipping.")
        return

    print(f"\nProcessing '{input_filename}'...")

    # Load the table, reading all data as strings to be safe
    df = pd.read_csv(input_filename, dtype=str)

    # --- Step 1: Prepare text data ---
    # Fill any missing values in text columns with an empty string
    text_columns = ['title', 'description']
    for col in text_columns:
        if col not in df.columns:
            df[col] = ''
        else:
            df[col].fillna('', inplace=True)

    # Combine relevant text columns for dense embedding
    print("Creating combined text for dense embeddings...")
    df["title"] = df["title"].str.lower()
    df["brand"] = df["brand"].str.lower()
    df["description"] = df["description"].str.lower()
    df['descr_shortened'] = df['description'].str[:120]
    df['combined_text'] =  df['title'] + ' ' + df['description']

    # --- Step 2: Generate Dense Embeddings (MiniLM) ---
    print("Generating dense embeddings... (This may take a while)")
    sentences = df['combined_text'].tolist()
    embeddings = model.encode(sentences, show_progress_bar=True)

    # Add the embeddings to the DataFrame in a new column 'v'
    df['v'] = [emb.tolist() for emb in embeddings]

    # --- Step 3: Generate MinHash Vectors ---
    print("Generating MinHash vectors for title (mv1)...")
    df['mv1'] = df['title'].apply(get_minhash_vector)

    print("Generating MinHash vectors for description (mv2)...")
    df['mv2'] = df['description'].apply(get_minhash_vector)

    # --- Step 4: Finalize and Save ---
    # Drop the temporary combined_text column
    df.drop(columns=['combined_text'], inplace=True)

    # Save the updated DataFrame to a new CSV file
    df.to_parquet(output_filename, engine="pyarrow")
    print(f"Successfully created '{output_filename}' with new feature columns.")


if __name__ == '__main__':
    # --- Setup ---
    # Specify the pre-trained model for sentence embeddings
    model_name = 'all-MiniLM-L6-v2'
    model_name = "all-mpnet-base-v2"
    #model_name="wdc-finetuned-minilm"
    model_path = './data/wdc/wdc-finetuned-minilm'
    # Load the model. The library handles loading all the necessary files from that folder.
    #print(f"Loading fine-tuned model from: {model_name}")

    #print(f"Loading sentence transformer model: '{model_name}'...")
    embedding_model = SentenceTransformer(model_name)
    #embedding_model = SentenceTransformer(model_name)

    # --- Processing ---
    # Process tableA
    process_and_embed_table(
        input_filename='./data/wdc/tableA_.csv',
        output_filename='./data/wdc/tableA_.pqt',
        model=embedding_model
    )

    # Process tableB
    process_and_embed_table(
        input_filename='./data/wdc/tableB_.csv',
        output_filename='./data/wdc/tableB_.pqt',
        model=embedding_model
    )

    print("\nFeature generation complete for both tables.")
