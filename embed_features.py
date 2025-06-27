import pandas as pd
import numpy as np
import os
import re
from sentence_transformers import SentenceTransformer
import warnings
import mmh3  # Added for your custom MinHash function
from transformers import pipeline

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


print("Loading Question-Answering model...")
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
print("Model loaded.")

# Define a set of questions to probe the description for key features
PROBING_QUESTIONS = [
    #"What is the key feature?",
    "What is the model number or part number?",
    "What is the capacity or size?",
    "What is the resolution?",
    "What is the material?",
    "What is the color?",
    "What is the technology used?"
]


def summarize_description(description):
    """
    Uses a QA model to extract key features from a description.
    """
    if not isinstance(description, str) or len(description) < 10:
        return description  # Return empty if description is too short or not a string

    feature_answers = []
    # Ask each question and collect the answers
    for question in PROBING_QUESTIONS:
        result = qa_pipeline(question=question, context=description)
        # We only keep high-confidence answers to avoid adding noise
        if result['score'] > 0.1:
            feature_answers.append(result['answer'])

    # Return a unique, space-separated string of the found features
    return " ".join(sorted(list(set(feature_answers))))


def serialize_row(row, columns_to_serialize):
    """
    Serializes specified columns of a DataFrame row into a single string.
    """
    serialized_parts = []
    for col in columns_to_serialize:
        value = row.get(col, '')

        # If the column is 'description', we use our new summarizer
        if col == 'description':
            processed_value = summarize_description(value)
        else:
            # For other columns like title and brand, we just clean them
            processed_value = str(value).lower().strip()

        if processed_value:
            serialized_parts.append(f"[COL] {col} [VAL] {processed_value}")

    return " ".join(serialized_parts)



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
    ##df["title"] = df["title"].str.lower()
    #df["brand"] = df["brand"].str.lower()
    #df["description"] = df["description"].str.lower()

    #columns_to_serialize = ['brand', 'title', 'description']
    columns_to_serialize = ['title','description']

    # --- 3. Apply the serialization function to create the new column ---
    # We use .apply() with a lambda function to pass the list of columns
    # to our main serialization function for each row.
    #df['combined_text'] = df.apply(
    ##    lambda row: serialize_row(row, columns_to_serialize),
    #    axis=1
    #)
    #df["description"] = summarize_description(df["description"].str.lower())  # df["description"].str.lower()

    df['combined_text'] =  df['title'] + ' ' + df['description']

    # --- Step 2: Generate Dense Embeddings (MiniLM) ---
    print("Generating dense embeddings... (This may take a while)")
    sentences = df['combined_text'].tolist()
    embeddings = model.encode(sentences, show_progress_bar=True)

    # Add the embeddings to the DataFrame in a new column 'v'
    df['v'] = [emb.tolist() for emb in embeddings]

    # --- Step 3: Generate MinHash Vectors ---
    #print("Generating MinHash vectors for title (mv1)...")
    #df['mv1'] = df['title'].apply(get_minhash_vector)

    #print("Generating MinHash vectors for description (mv2)...")
    #df['mv2'] = df['description'].apply(get_minhash_vector)

    # --- Step 4: Finalize and Save ---
    # Drop the temporary combined_text column
    df.drop(columns=['combined_text'], inplace=True)

    # Save the updated DataFrame to a new CSV file
    df.to_parquet(output_filename, engine="pyarrow")

    print(f"Successfully created '{output_filename}' with new feature columns.")

