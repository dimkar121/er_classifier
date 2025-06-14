import pandas as pd
import faiss
import numpy as np
import math
from sentence_transformers import InputExample
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader
import time

def fine_tune():
    df1 = pd.read_parquet(f"./data/wdc/tableA_train.pqt")
    df2 = pd.read_parquet(f"./data/wdc/tableB_train.pqt")
    gold_standard = pd.read_csv(f"./data/wdc/gold_standard_train.csv", sep=",", encoding="utf-8", keep_default_na=False)
    df1['id'] = pd.to_numeric(df1['id'])
    df2['id'] = pd.to_numeric(df2['id'])
    a_embeddings = df1['v'].tolist()
    b_embeddings = df2['v'].tolist()
    d = 768
    faiss_db = faiss.IndexHNSWFlat(d, 32)
    faiss_db.hnsw.efConstruction = 60
    faiss_db.hnsw.efSearch = 16
    datav = np.array(a_embeddings).astype(np.float32)
    faiss_db.add(datav)
    ids1_ = df1['id'].tolist()


    training_triplets_ids = []
    k = 10  # Number of nearest neighbors to search for

    for index, row in gold_standard.iterrows():
        b_id = row['id_B']
        positive_a_id = row['id_A']

        # Get the index (row number) of the Google record
        anchor_idx = df2[df2['id'] == b_id].index[0]

        # Get the corresponding embedding for the anchor (Google product)
        anchor_embedding = b_embeddings[anchor_idx].reshape(1, -1)

        # Search FAISS for the k nearest neighbors in the Amazon dataset
        distances, indices = faiss_db.search(anchor_embedding, k)

        # The result 'indices' is a 2D array, so we take the first row
        neighbor_a_indices = indices[0]

        # Find the first neighbor that is NOT the true positive match
        hard_negative_a_ids = []
        for a_idx in neighbor_a_indices:
            # Get the actual ID from the index
            potential_neg_id = df1.iloc[a_idx]['id']

            if potential_neg_id != positive_a_id:
                hard_negative_a_ids.append(potential_neg_id)


            if len(hard_negative_a_ids)==2:
                break  # We found our hard negatives, so we can stop searching

        easy_negative_a_ids = []
        while True:
            # Select a completely random product from the buy dataset
            random_a_record = df1.sample(1).iloc[0]
            # Make sure it's not the actual positive match
            if random_a_record['id'] != positive_a_id:
                easy_negative_a_ids.append(random_a_record['id'])

            if len(easy_negative_a_ids) == 2:
                 break


        # If we found a valid hard negative, store the triplet of IDs
        if hard_negative_a_ids and easy_negative_a_ids:
            for hard_negative_a_id in hard_negative_a_ids:
               training_triplets_ids.append({
                 'b_id': b_id,
                 'positive_a_id': positive_a_id,
                 'negative_a_id': hard_negative_a_id
               })
            for easy_negative_a_id in easy_negative_a_ids:
              training_triplets_ids.append({
                'b_id': b_id,
                'positive_a_id': positive_a_id,
                'negative_a_id': easy_negative_a_id,
                'type': 'easy'
              })

    print(f"Found {len(training_triplets_ids)} triplets.")

    # --- Step 3: Retrieve Text and Create InputExamples ---
    print("\n--- 3. Creating InputExample objects for training ---")
    train_examples = []
    # Create mapping dictionaries for fast text lookup
    df1['descr_shortened'] = df1['description'].str[:120]
    df1['combined_text'] = df1['title'] #+ ' ' + df1['description']
    df2['descr_shortened'] = df2['description'].str[:120]
    df2['combined_text'] = df2['title'] #+ ' ' + df2['description']
    a_id_to_text = pd.Series(df1.combined_text.values, index=df1.id).to_dict()
    b_id_to_text = pd.Series(df2.combined_text.values, index=df2.id).to_dict()

    for triplet in training_triplets_ids:
        anchor_text =   b_id_to_text.get(triplet['b_id'])
        positive_text = a_id_to_text.get(triplet['positive_a_id'])
        negative_text = a_id_to_text.get(triplet['negative_a_id'])

        if anchor_text and positive_text and negative_text:
            train_examples.append(InputExample(texts=[anchor_text, positive_text, negative_text]))

    print(f"Final number of training examples: {len(train_examples)}")
    print("\nExample of a training triplet:")

    for t  in range(100):
       print(train_examples[t].texts)

    # Load the pre-trained MiniLM model that we will fine-tune
    model_name = 'all-MiniLM-L6-v2'
    model_name = "all-mpnet-base-v2"
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
    output_model_path = './data/wdc/wdc-finetuned-model'  # The path where the new model will be saved

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












