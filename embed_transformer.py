import pandas as pd
from sentence_transformers import SentenceTransformer
import os
import torch
import pickle
import utilities

def embed(df, text_columns, prefix, output_filename, model, name_minhash=None):
    """
    Loads a table, generates dense embeddings and MinHash vectors,
    and saves the result to a new file.
    """
    # if not os.path.exists(input_filename):
    #    print(f"Error: Input file '{input_filename}' not found. Skipping.")
    #    return

    # print(f"\nProcessing '{input_filename}'...")

    # --- Step 1: Prepare text data ---
    # Fill any missing values in text columns with an empty string

    #df['combined_text'] = prefix + df[text_columns].fillna('').apply(lambda row: ' '.join(row), axis=1)
    df['combined_text'] = prefix + df[text_columns].astype(str).agg(' '.join, axis=1)
    print(df['combined_text'])

    # Combine relevant text columns for dense embedding
    print("Creating combined text for dense embeddings...")

    # --- Step 2: Generate Dense Embeddings (MiniLM) ---
    print("Generating dense embeddings... (This may take a while)")
    sentences = df['combined_text'].tolist()
    embeddings = model.encode(sentences, show_progress_bar=True)

    # Add the embeddings to the DataFrame in a new column 'v'
    df['v'] = [emb.tolist() for emb in embeddings]
 
    if name_minhash is not None:
         df[f"{name_minhash}_minhash"] = df[name_minhash].apply(utilities.get_minhash)   
         df[f"{name_minhash}_bytes"] = df[f"{name_minhash}_minhash"].apply(pickle.dumps)
         df = df.drop(columns=[f"{name_minhash}_minhash"])


    # --- Step 3: Generate MinHash Vectors ---
    # print("Generating MinHash vectors for title (mv1)...")
    # df['mv1'] = df['title'].apply(get_minhash_vector)

    # print("Generating MinHash vectors for description (mv2)...")
    # df['mv2'] = df['description'].apply(get_minhash_vector)

    # --- Step 4: Finalize and Save ---
    # Drop the temporary combined_text column
    df.drop(columns=['combined_text'], inplace=True)

    # Save the updated DataFrame to a new CSV file
    df.to_parquet(output_filename, engine="pyarrow")

    print(f"Successfully created '{output_filename}' with new feature columns.")


if __name__ == '__main__':
    # This block runs when the script is executed directly
    model_name = 'all-MiniLM-L6-v2'
    #model_name = 'intfloat/e5-large-v2'
    #model_name = "all-mpnet-base-v2"
    # model_name = "roberta-base-nli-stsb-mean-tokens"

    folder = "./data"
    model_tag = "mini"
    embedding_model = SentenceTransformer(model_name)
    device_name = embedding_model.device.type
    if device_name == 'cuda':
        print(f"The model is on the GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"The model is on the CPU.")

    df = pd.read_csv("./data/Scholar.csv", sep=",", encoding="utf-8")
    embed(
        df=df,
        text_columns=["title","authors","venue","year"],
        prefix="",
        output_filename=f'{folder}/Scholar_{model_tag}.pqt',
        model=embedding_model, 
        name_minhash="title"
    )

    df = pd.read_csv("./data/DBLP2.csv", sep=",", encoding="utf-8")
    embed(
        df=df,
        text_columns=["title","authors","venue","year"],
        prefix="",
        output_filename=f'{folder}/DBLP2_{model_tag}.pqt',
        model=embedding_model,
        name_minhash="title"
    )









