import pandas as pd
from sentence_transformers import SentenceTransformer
import os

def embed(input_filename,text_columns, prefix, output_filename, model):
    """
    Loads a table, generates dense embeddings and MinHash vectors,
    and saves the result to a new file.
    """
    if not os.path.exists(input_filename):
        print(f"Error: Input file '{input_filename}' not found. Skipping.")
        return

    print(f"\nProcessing '{input_filename}'...")

    # Load the table, reading all data as strings to be safe
    df = pd.read_csv(input_filename, sep=",", encoding="utf-8", dtype=str)

    # --- Step 1: Prepare text data ---
    # Fill any missing values in text columns with an empty string

    df['combined_text'] = prefix + df[text_columns].fillna('').apply(lambda row: ' '.join(row), axis=1)        
    print(df['combined_text'])

    # Combine relevant text columns for dense embedding
    print("Creating combined text for dense embeddings...")


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



if __name__ == '__main__':
    # This block runs when the script is executed directly
    model_name = 'all-MiniLM-L6-v2'
    model_name = 'intfloat/e5-large-v2'
    #model_name = "all-mpnet-base-v2"
    #model_name = "roberta-base-nli-stsb-mean-tokens"

    folder = "./data"
    embedding_model = SentenceTransformer(model_name)
    #embed_features.process_and_embed_table(
    #    input_filename=f'{folder}/tableA.csv',
    #    output_filename=f'{folder}/tableA.pqt',
    #    model=embedding_model
    #)

    '''
    embed(
        input_filename=f'{folder}/DBLP2.csv',
        text_columns= ["title", "authors", "venue", "year"],
        prefix="academic publication: ", 
        output_filename=f'{folder}/DBLP2_e5.pqt',
        model=embedding_model
    )
    '''
    embed(
          input_filename=f'{folder}/Scholar.csv',
          text_columns= ["title", "authors", "venue", "year"],
          prefix="academic publication: ",
          output_filename=f'{folder}/Scholar_e5.pqt',
          model=embedding_model
    )









