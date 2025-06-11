import pandas as pd
import faiss
import numpy as np
import math
from sentence_transformers import InputExample
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader
import time
if __name__ == '__main__':

    df1 = pd.read_parquet(f"./data/Amazon_embedded_mini.pqt")
    df2 = pd.read_parquet(f"./data/Google_embedded_mini.pqt")
    gold_standard = pd.read_csv(f"./data/truth_Amazon_googleProducts.csv", sep=",", encoding="utf-8", keep_default_na=False)

    amazon_embeddings = df1['v'].tolist()
    google_embeddings = df2['v'].tolist()
    d = 384
    faiss_db = faiss.IndexHNSWFlat(d, 32)
    faiss_db.hnsw.efConstruction = 60
    faiss_db.hnsw.efSearch = 16
    datav = np.array(amazon_embeddings).astype(np.float32)
    faiss_db.add(datav)
    ids1_ = df1['id'].tolist()


    training_triplets_ids = []
    k = 10  # Number of nearest neighbors to search for

    for index, row in gold_standard.iterrows():
        google_id = row['idGoogleBase']
        positive_amazon_id = row['idAmazon']

        # Get the index (row number) of the Google record
        anchor_idx = df2[df2['id'] == google_id].index[0]

        # Get the corresponding embedding for the anchor (Google product)
        anchor_embedding = google_embeddings[anchor_idx].reshape(1, -1)

        # Search FAISS for the k nearest neighbors in the Amazon dataset
        distances, indices = faiss_db.search(anchor_embedding, k)

        # The result 'indices' is a 2D array, so we take the first row
        neighbor_amazon_indices = indices[0]

        # Find the first neighbor that is NOT the true positive match
        hard_negative_amazon_ids = []
        for amazon_idx in neighbor_amazon_indices:
            # Get the actual ID from the index
            potential_neg_id = df1.iloc[amazon_idx]['id']

            if potential_neg_id != positive_amazon_id:
                hard_negative_amazon_ids.append(potential_neg_id)


            if len(hard_negative_amazon_ids)==3:
                break  # We found our hard negatives, so we can stop searching

        easy_negative_amazon_ids = []
        while True:
            # Select a completely random product from the buy dataset
            random_amazon_record = df1.sample(1).iloc[0]
            # Make sure it's not the actual positive match
            if random_amazon_record['id'] != positive_amazon_id:
                easy_negative_amazon_ids.append(random_amazon_record['id'])

            if len(easy_negative_amazon_ids) == 3:
                 break


        # If we found a valid hard negative, store the triplet of IDs
        if hard_negative_amazon_ids and easy_negative_amazon_ids:
            for hard_negative_amazon_id in hard_negative_amazon_ids:
               training_triplets_ids.append({
                 'google_id': google_id,
                 'positive_amazon_id': positive_amazon_id,
                 'negative_amazon_id': hard_negative_amazon_id
               })
            for easy_negative_amazon_id in easy_negative_amazon_ids:
              training_triplets_ids.append({
                'google_id': google_id,
                'positive_amazon_id': positive_amazon_id,
                'negative_amazon_id': easy_negative_amazon_id,
                'type': 'easy'
              })

    print(f"Found {len(training_triplets_ids)} triplets.")

    # --- Step 3: Retrieve Text and Create InputExamples ---
    print("\n--- 3. Creating InputExample objects for training ---")
    train_examples = []
    # Create mapping dictionaries for fast text lookup
    amazon_id_to_text = pd.Series(df1.name.values, index=df1.id).to_dict()
    google_id_to_text = pd.Series(df2.name.values, index=df2.id).to_dict()

    for triplet in training_triplets_ids:
        anchor_text =   google_id_to_text.get(triplet['google_id'])
        positive_text = amazon_id_to_text.get(triplet['positive_amazon_id'])
        negative_text = amazon_id_to_text.get(triplet['negative_amazon_id'])

        if anchor_text and positive_text and negative_text:
            train_examples.append(InputExample(texts=[anchor_text, positive_text, negative_text]))

    print(f"Final number of training examples: {len(train_examples)}")
    print("\nExample of a training triplet:")
    print(train_examples[0].texts)

    # Load the pre-trained MiniLM model that we will fine-tune
    model_name = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)

    # TripletLoss requires a dataloader that creates smart batches.
    # We'll use a batch size of 32, but you can adjust this based on your GPU memory.
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)

    # Define the TripletLoss function. This is the heart of the fine-tuning.
    train_loss = losses.TripletLoss(model=model)

    print("--- Model, Loss Function, and DataLoader are Ready ---")

    # --- 2. Fine-Tune the Model ---

    # Configure the training
    num_epochs = 1  # 1-4 epochs is usually sufficient for fine-tuning on this task.
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)  # 10% of training steps for warm-up
    output_model_path = './data/amazon-google-finetuned-minilm'  # The path where the new model will be saved

    # The model's .fit() method orchestrates the entire training process
    print("\n--- Starting the Fine-Tuning Process ---")
    print(f"This will run for {num_epochs} epochs.")
    st = time.time()
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=output_model_path,
        show_progress_bar=True,
        checkpoint_path=f'{output_model_path}/checkpoints/',  # Save checkpoints
        checkpoint_save_steps=500  # Save a checkpoint every 500 steps
    )
    end =  time.time()

    # --- 3. Confirmation ---

    print(f"\n--- Fine-Tuning Complete! --- in {end-st} seconds.")
    print(f"Your new, specialized model has been saved to: '{output_model_path}'")












