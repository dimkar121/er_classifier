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
import embed_transformer
from sentence_transformers.evaluation import TripletEvaluator

folder = "./data"


def fine_tune(text_columns_1, text_columns_2, gold_standard, id1, id2, pq1, pq2, model_name, model_path, device="cuda"):
    print(f"model to use {model_name}")  
    a_embeddings = pq1['v'].tolist()
    b_embeddings = pq2['v'].tolist()
    datav = np.array(a_embeddings).astype(np.float32)
    d = datav.shape[1]
    faiss_db = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    faiss_db.hnsw.efConstruction = 60
    faiss_db.hnsw.efSearch = 16
    faiss_db.add(datav)
    ids1_ = pq1['id'].tolist()

    training_triplets_ids = []
    k = 10  # Number of nearest neighbors to search for

    for index, row in gold_standard.iterrows():
        b_id = row[id2]
        positive_a_id = row[id1]

        if not positive_a_id in pq1['id'].values or not b_id in pq2['id'].values:
            continue

   
        # Get the index (row number) of the Google record
        anchor_idx = pq2[pq2['id'] == b_id].index[0]
    
        # Get the corresponding embedding for the anchor (Google product)
        anchor_embedding = b_embeddings[anchor_idx].reshape(1, -1).astype(np.float32)

        # Search FAISS for the k nearest neighbors in the Amazon dataset
        distances, indices = faiss_db.search(anchor_embedding, k)

        # The result 'indices' is a 2D array, so we take the first row
        neighbor_a_indices = indices[0]

        # Find the first neighbor that is NOT the true positive match
        hard_negative_a_ids = []
        for a_idx in neighbor_a_indices:
            # Get the actual ID from the index
            potential_neg_id = pq1.iloc[a_idx]['id']

            if potential_neg_id != positive_a_id:
                hard_negative_a_ids.append(potential_neg_id)

            if len(hard_negative_a_ids) == 2:
                break  # We found our hard negatives, so we can stop searching

        easy_negative_a_ids = []
        while True:
            # Select a completely random product from the buy dataset
            random_a_record = pq1.sample(1).iloc[0]
            # Make sure it's not the actual positive match
            if random_a_record['id'] != positive_a_id:
                easy_negative_a_ids.append(random_a_record['id'])

            if len(easy_negative_a_ids) == 1:
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

    #pq1['combined_text'] = pq1[text_columns_1].fillna('').apply(lambda row: ' '.join(row), axis=1)
    pq1['combined_text'] = pq1[text_columns_1].astype(str).agg(' '.join, axis=1)
    pq2['combined_text'] = pq2[text_columns_2].astype(str).agg(' '.join, axis=1)
    #pq2['combined_text'] = pq2[text_columns_2].fillna('').apply(lambda row: ' '.join(row), axis=1)
    a_id_to_text = pd.Series(pq1.combined_text.values, index=pq1.id).to_dict()
    b_id_to_text = pd.Series(pq2.combined_text.values, index=pq2.id).to_dict()

    from sklearn.model_selection import train_test_split

    # Split your single list into two new lists
    train_triplets, eval_triplets = train_test_split(
      training_triplets_ids,
      test_size=0.1,         # Use 10% of the data for evaluation
      random_state=42        # Ensures the split is the same every time
    )


    sentences1 = []
    sentences2 = []
    labels = []
    #for triplet in training_triplets_ids:
    for triplet in train_triplets:
        anchor_text = b_id_to_text.get(triplet['b_id'])
        positive_text = a_id_to_text.get(triplet['positive_a_id'])
        negative_text = a_id_to_text.get(triplet['negative_a_id'])

        if anchor_text and positive_text and negative_text:
            train_examples.append(
                InputExample(texts=[anchor_text, positive_text, negative_text]))  # this is for tripletloss

            sentences1.append(anchor_text)
            sentences2.append(positive_text)
            labels.append(1)

            # Negative pair
            sentences1.append(anchor_text)
            sentences2.append(negative_text)
            labels.append(0)

    print(f"Final number of training examples: {len(train_examples)}")
    print("\nExample of a training triplet:")

    for t in range(100):
        print(train_examples[t].texts)

    

    # Create empty lists for the evaluator
    eval_anchors = []
    eval_positives = []
    eval_negatives = []

    # It's crucial to use a separate evaluation set, NOT your training_triplets_ids
    for triplet in eval_triplets:
      anchor_text = b_id_to_text.get(triplet['b_id'])
      positive_text = a_id_to_text.get(triplet['positive_a_id'])
      negative_text = a_id_to_text.get(triplet['negative_a_id'])

      if  anchor_text and positive_text and negative_text:
         eval_anchors.append(anchor_text)
         eval_positives.append(positive_text)
         eval_negatives.append(negative_text)



    model = SentenceTransformer(model_name)

    #model.to(device)

    # TripletLoss requires a dataloader that creates smart batches.
    # We'll use a batch size of 32, but you can adjust this based on your GPU memory.
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)

    # Define the TripletLoss function. This is the heart of the fine-tuning.
    train_loss = losses.TripletLoss(model=model)
    # train_loss = losses.SoftmaxLoss(
    #    model=model,
    #    sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
    #    num_labels=2  # binary classification (match / non-match)
    # )

    print("--- Model, Loss Function, and DataLoader are Ready ---")

    # --- 2. Fine-Tune the Model ---

    # Configure the training
    num_epochs = 1  # 1-4 epochs is usually sufficient for fine-tuning on this task.
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)  # 10% of training steps for warm-up
    output_model_path = model_path  # The path where the new model will be saved

    #evaluator = evaluation.BinaryClassificationEvaluator(
    #    sentences1=sentences1,
    #    sentences2=sentences2,
    #    labels=labels  # 1 = match, 0 = non-match
    #)

    triplet_evaluator = TripletEvaluator(eval_anchors, eval_positives, eval_negatives)
    from torch.optim import AdamW
    # The model's .fit() method orchestrates the entire training process
    print("\n--- Starting the Fine-Tuning Process ---")
    print(f"This will run for {num_epochs} epochs.")
    st = time.time()
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        evaluation_steps=50,
        evaluator=triplet_evaluator,
        warmup_steps=warmup_steps,
        output_path=output_model_path,
        show_progress_bar=True,
        checkpoint_path=f'{output_model_path}/checkpoints/',  # Save checkpoints
        checkpoint_save_steps=500,  # Save a checkpoint every 500 steps
        #use_amp=False  # <-- Add this line
        optimizer_class=AdamW,
        optimizer_params={'lr': 2e-5}
    )
    end = time.time()

    # --- 3. Confirmation ---

    print(f"\n--- Fine-Tuning Complete! --- in {end - st} seconds.")
    print(f"Your new, specialized model has been saved to: '{output_model_path}'")


import torch
import random
if __name__ == '__main__':
    # This block runs when the script is executed directly
    #model_name = 'all-MiniLM-L6-v2'
    model_name = "all-mpnet-base-v2"
    # model_name = "roberta-base-nli-stsb-mean-tokens"
    #model_name = 'intfloat/e5-large-v2'

    #embedding_model = SentenceTransformer(model_name)
    model_tag = "mpnet"
    
    
    pq1 = pd.read_parquet(f"./data/Abt_{model_tag}.pqt")
    #pq1 = pq1.dropna(subset=['title'])
    pq2 = pd.read_parquet(f"./data/Buy_{model_tag}.pqt")
    gold_standard = pd.read_csv(f"./data/truth_abt_buy.csv", sep=",", encoding="utf-8", keep_default_na=False)

    #pq1['id'] = pd.to_numeric(pq1['id'], errors='coerce')
    #pq2['id'] = pd.to_numeric(pq2['id'], errors='coerce')
    #valid_d1_ids = set(pq1['id'].values)
    #valid_d2_ids = set(pq2['id'].values)
    #mask_to_keep = gold_standard['D1'].isin(valid_d1_ids) & gold_standard['D2'].isin(valid_d2_ids)
    #gold_standard = gold_standard[mask_to_keep].copy()
    #matches = len(gold_standard.keys()) 


    #print(pq1.dtypes)
    #print(pq2.dtypes)
    #print( gold_standard.dtypes)
    #exit(1)

    #pq2['id'] = pd.to_numeric(pq2['id'], errors='coerce')
    #pq2.dropna(subset=['id'], inplace=True)
    #pq2['id'] = pq2['id'].astype(int)
    #pq1['id'] = pd.to_numeric(pq1['id'], errors='coerce')
    #pq1.dropna(subset=['id'], inplace=True)
    #pq1['id'] = pq1['id'].astype(int)    
    #gold_standard['id1'] = gold_standard['id1'].astype(int)
    #gold_standard['id2'] = gold_standard['id2'].astype(int)
    #pq2.reset_index(drop=True, inplace=True)
    #pq1.reset_index(drop=True, inplace=True)


    model_path = f'./data/abt_buy_{model_tag}_ft_model'
    fine_tune(["name","description","price"],
              ["name","description","price"],
              gold_standard, "idAbt", "idBuy", pq1, pq2, model_name, model_path)

    df = pd.read_csv("./data/Abt.csv", sep=",", encoding="unicode_escape")
    embedding_model = SentenceTransformer(model_path)
    embed_transformer.embed(
        df=df,
        text_columns=["name","description","price"],
        prefix="",
        output_filename=f'./data/Abt_{model_tag}_ft.pqt',
        model=embedding_model,
        name_minhash="name",
    )

    df = pd.read_csv("./data/Buy.csv", sep=",", encoding="unicode_escape")
    embed_transformer.embed(
        df=df,
        text_columns=["name","description","price"],
        prefix="",
        output_filename=f'./data/Buy_{model_tag}_ft.pqt',
        model=embedding_model,       
        name_minhash="name",
    )
    
