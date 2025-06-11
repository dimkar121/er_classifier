import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


if __name__ == '__main__':
   
   df1 = pd.read_csv("./data/Abt.csv", sep=",", encoding="unicode_escape", keep_default_na=False)
   df2 = pd.read_csv("./data/Buy.csv", sep=",", encoding="unicode_escape", keep_default_na=False)
   model_path = './data/abt-buy-finetuned-minilm'
   # Load the model. The library handles loading all the necessary files from that folder.
   print(f"Loading fine-tuned model from: {model_path}")
   model = SentenceTransformer(model_path)
   print("Model loaded successfully!")

   abt_names_corpus = df1['name'].fillna('').tolist()
   buy_names_corpus = df2['name'].fillna('').tolist()

   print(f"\nFound {len(abt_names_corpus)} product names in the Abt dataset.")
   print(f"Found {len(buy_names_corpus)} product names in the Buy dataset.")

   # --- Generate the new embeddings IN BATCHES ---
   print("\nGenerating new embeddings for the Abt dataset... (This may take a moment)")
   # Set a batch size that works for your hardware (e.g., 32, 64, 128)
   abt_embeddings_finetuned = model.encode(
       abt_names_corpus,
       batch_size=32,
       show_progress_bar=True,
       convert_to_numpy=True
   )

   print("\nGenerating new embeddings for the Buy dataset...")
   buy_embeddings_finetuned = model.encode(
       buy_names_corpus,
       batch_size=32,
       show_progress_bar=True,
       convert_to_numpy=True
   )

   print("\n--- Embedding Generation Complete! ---")
   print(f"Shape of new Abt embeddings: {abt_embeddings_finetuned.shape}")
   print(f"Shape of new Buy embeddings: {buy_embeddings_finetuned.shape}")

   df1['v'] = list(abt_embeddings_finetuned)
   df2['v'] = list(buy_embeddings_finetuned)
   print(df1.head())
   df1.to_parquet("./data/Abt_embedded_mini_ft.pqt", engine='pyarrow')
   df2.to_parquet("./data/Buy_embedded_mini_ft.pqt", engine='pyarrow')

 



