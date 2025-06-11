import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


if __name__ == '__main__':
   
   df1 = pd.read_csv("./data/Amazon.csv", sep=",", encoding="unicode_escape", keep_default_na=False)
   df2 = pd.read_csv("./data/GoogleProducts.csv", sep=",", encoding="unicode_escape", keep_default_na=False)
   model_path = './data/amazon-google-finetuned-minilm'
   # Load the model. The library handles loading all the necessary files from that folder.
   print(f"Loading fine-tuned model from: {model_path}")
   model = SentenceTransformer(model_path)
   print("Model loaded successfully!")

   amazon_names_corpus = df1['title'].fillna('').tolist()
   google_names_corpus = df2['name'].fillna('').tolist()

   print(f"\nFound {len(amazon_names_corpus)} product names in the Abt dataset.")
   print(f"Found {len(google_names_corpus)} product names in the Buy dataset.")

   # --- Generate the new embeddings IN BATCHES ---
   print("\nGenerating new embeddings for the Amazon dataset... (This may take a moment)")
   # Set a batch size that works for your hardware (e.g., 32, 64, 128)
   amazon_embeddings_finetuned = model.encode(
       amazon_names_corpus,
       batch_size=32,
       show_progress_bar=True,
       convert_to_numpy=True
   )

   print("\nGenerating new embeddings for the Google dataset...")
   google_embeddings_finetuned = model.encode(
       google_names_corpus,
       batch_size=32,
       show_progress_bar=True,
       convert_to_numpy=True
   )

   print("\n--- Embedding Generation Complete! ---")
   print(f"Shape of new Amazon embeddings: {amazon_embeddings_finetuned.shape}")
   print(f"Shape of new Google embeddings: {google_embeddings_finetuned.shape}")

   df1['v'] = list(amazon_embeddings_finetuned)
   df2['v'] = list(google_embeddings_finetuned)
   print(df1.head())
   df1.to_parquet("./data/Amazon_embedded_mini_ft.pqt", engine='pyarrow')
   df2.to_parquet("./data/Google_embedded_mini_ft.pqt", engine='pyarrow')

 



