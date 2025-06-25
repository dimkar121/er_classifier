import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sentence_transformers import InputExample
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader
from sentence_transformers import evaluation
import time
import utilities


folder="./data"



def embed(input_filename, output_filename, text_columns_embed,text_columns_minhash, model):
    """
    Loads a table, generates dense embeddings and MinHash vectors,
    and saves the result to a new file.
    """
    if not os.path.exists(input_filename):
        print(f"Error: Input file '{input_filename}' not found. Skipping.")
        return

    print(f"\nProcessing '{input_filename}'...")

    # Load the table, reading all data as strings to be safe
    df = pd.read_csv(input_filename, sep=",",  encoding="unicode_escape", keep_default_na=False)
    print(f"Number of rows:{len(df)}")

    # --- Step 1: Prepare text data ---
    # Fill any missing values in text columns with an empty string

    for col in text_columns_embed:
        if col not in df.columns:
            df[col] = ''
        else:
            df[col].fillna('', inplace=True)

    # Combine relevant text columns for dense embedding
    print("Creating combined text for dense embeddings...")


    #df['combined_text'] = df[text_columns].fillna('').sum(axis=1)
    df['combined_text'] = df[text_columns_embed].fillna('').apply(lambda row: ' '.join(row), axis=1)

    print("Generating dense embeddings... (This may take a while)")
    sentences = df['combined_text'].tolist()
    embeddings = model.encode(sentences, show_progress_bar=True)

    # Add the embeddings to the DataFrame in a new column 'v'
    df['v'] = [emb.tolist() for emb in embeddings]

    for col in text_columns_minhash:
        df[col] = df[col].str.lower()

    for col in text_columns_minhash:
        df[f"{col}_v"] =  df[col].apply(lambda x: utilities.minhash(utilities.bigrams(x) ) )

    # --- Step 3: Generate MinHash Vectors ---
    #print("Generating MinHash vectors for title (mv1)...")
    #df['mv1'] = df['title'].apply(get_minhash_vector)

    #print("Generating MinHash vectors for description (mv2)...")
    #df['mv2'] = df['description'].apply(get_minhash_vector)

    # --- Step 4: Finalize and Save ---
    # Drop the temporary combined_text column
    df.drop(columns=['combined_text'], inplace=True)
    # Save the updated DataFrame to a new CSV file
    print(f"Number of rows:{len(df)}")
    df.to_parquet(output_filename, engine="pyarrow")

    print(f"Successfully created '{output_filename}' with new feature columns.")





if __name__ == '__main__':
    # This block runs when the script is executed directly
    model_name = 'all-MiniLM-L6-v2'
    #model_name = "all-mpnet-base-v2"
    #model_name = "roberta-base-nli-stsb-mean-tokens"

    #embedding_model = SentenceTransformer(model_name)
    embedding_model = SentenceTransformer("./data/amazon-google-finetuned-minilm")
    embed(
        input_filename=f'{folder}/Amazon.csv',
        output_filename=f'{folder}/Amazon_mini_ft.pqt',
        text_columns_embed=["title","description", "manufacturer"],
        text_columns_minhash=["title", "description", "manufacturer"],
        model=embedding_model
    )
    embed(
        input_filename=f'{folder}/GoogleProducts.csv',
        output_filename=f'{folder}/GoogleProducts_mini_ft.pqt',
        text_columns_embed=["name","description", "manufacturer"],
        text_columns_minhash=["name" ,"description","manufacturer"],
        model=embedding_model
    )

