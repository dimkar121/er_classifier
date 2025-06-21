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

def fine_tune(text_columns_walmart, text_columns_amazon):
    df2 = pd.read_parquet(f"./data/walmart_products.pqt")
    df1 = pd.read_parquet(f"./data/amazon_products.pqt")
    #df2.dropna(subset=['id'], inplace=True)
    #df2['id'] = df2['id'].astype(int)

    gold_standard = pd.read_csv(f"./data/truth_amazon_walmart.tsv", sep="\t", encoding="utf-8", keep_default_na=False)
    gold_standard['id1'] = gold_standard['id1'].astype(int)
    gold_standard['id2'] = gold_standard['id2'].astype(int)

    mask_to_keep = pd.to_numeric(df1['id'], errors='coerce').notna()
    # 3. Apply the mask to the DataFrame to keep only the good rows.
    df_cleaned = df1[mask_to_keep].copy()
    # Optional: Now that all rows are clean, you can safely cast the 'id' column to integer.
    df_cleaned['id'] = df_cleaned['id'].astype(int)
    df_cleaned.to_parquet(f"./data/amazon_products.pqt")
    df1 = df_cleaned

    mask_to_keep = pd.to_numeric(df2['id'], errors='coerce').notna()
    # 3. Apply the mask to the DataFrame to keep only the good rows.
    df_cleaned = df2[mask_to_keep].copy()
    # Optional: Now that all rows are clean, you can safely cast the 'id' column to integer.
    df_cleaned['id'] = df_cleaned['id'].astype(int)
    df_cleaned.to_parquet(f"./data/walmart_products.pqt")
    df2 = df_cleaned

    '''
    error_mask = pd.to_numeric(df1['id'], errors='coerce').isna()
    # 3. Filter the original DataFrame using the mask to show only the problem rows
    problematic_rows = df1[error_mask]
    # 4. Print the results
    print("\n--- Rows causing the conversion error A---")
    if problematic_rows.empty:
        print("No problematic rows found.")
    else:
        print(problematic_rows)

    error_mask = pd.to_numeric(df2['id'], errors='coerce').isna()
    # 3. Filter the original DataFrame using the mask to show only the problem rows
    problematic_rows = df1[error_mask]
    # 4. Print the results
    print("\n--- Rows causing the conversion error B---")
    if problematic_rows.empty:
        print("No problematic rows found.")
    else:
        print(problematic_rows)
    exit()
    '''
    #df1 = pd.read_parquet(f"./data/walmart_products.pqt")
    #df2 = pd.read_parquet(f"./data/amazon_products.pqt")

    df1['id'] = pd.to_numeric(df1['id'], errors='coerce')
    df1.dropna(subset=['id'], inplace=True)
    df1['id'] = df1['id'].astype(int)

    df2['id'] = pd.to_numeric(df2['id'], errors='coerce')
    df2.dropna(subset=['id'], inplace=True)
    df2['id'] = df2['id'].astype(int)


    a_embeddings = df1['v'].tolist()
    b_embeddings = df2['v'].tolist()
    d = 384
    faiss_db = faiss.IndexHNSWFlat(d, 32)
    faiss_db.hnsw.efConstruction = 60
    faiss_db.hnsw.efSearch = 16
    datav = np.array(b_embeddings).astype(np.float32)
    faiss_db.add(datav)
    ids1_ = df1['id'].tolist()


    training_triplets_ids = []
    k = 15  # Number of nearest neighbors to search for
    '''
    main_id_set = set(df1['id'].values)
    gold_id_set = set(gold_standard['id2'].values)
    missing_ids = gold_id_set.difference(main_id_set)
    if not missing_ids:
        print("Success! All gold standard IDs are present in the main DataFrame.")
    else:
        print(f"Error: The following {len(missing_ids)} IDs from the gold standard were NOT found in df1:")
        print(list(missing_ids))

    exit()
    '''
    #gold_standard id1 -> amazon   gold_standard id2 -> walmart

    for index, row in gold_standard.iterrows():
        a_id = row['id1']
        positive_b_id = row['id2']

        if not positive_b_id in df2['id'].values or not a_id in df1['id'].values:
            print("Either of these is orphan", a_id, positive_b_id)
            continue

        # Get the index (row number) of the Google record
        anchor_idx = df1[df1['id'] == a_id].index[0]

        # Get the corresponding embedding for the anchor (Google product)

        if anchor_idx >= len(a_embeddings):
            print("Problem",anchor_idx, len(a_embeddings) )
            continue
        anchor_embedding = a_embeddings[anchor_idx].reshape(1, -1).astype(np.float32)

        # Search FAISS for the k nearest neighbors in the Amazon dataset
        distances, indices = faiss_db.search(anchor_embedding, k)

        # The result 'indices' is a 2D array, so we take the first row
        neighbor_b_indices = indices[0]

        # Find the first neighbor that is NOT the true positive match
        hard_negative_b_ids = []
        for b_idx in neighbor_b_indices:
            # Get the actual ID from the index
            potential_neg_id = df2.iloc[b_idx]['id']

            if potential_neg_id != positive_b_id:
                hard_negative_b_ids.append(potential_neg_id)


            if len(hard_negative_b_ids)==2:
                break  # We found our hard negatives, so we can stop searching

        easy_negative_b_ids = []
        while True:
            # Select a completely random product from the buy dataset
            random_b_record = df2.sample(1).iloc[0]
            # Make sure it's not the actual positive match
            if random_b_record['id'] != positive_b_id:
                easy_negative_b_ids.append(random_b_record['id'])

            if len(easy_negative_b_ids) == 1:
                 break


        # If we found a valid hard negative, store the triplet of IDs
        if hard_negative_b_ids and easy_negative_b_ids:
            for hard_negative_b_id in hard_negative_b_ids:
               training_triplets_ids.append({
                 'a_id': a_id,
                 'positive_b_id': positive_b_id,
                 'negative_b_id': hard_negative_b_id
               })
            for easy_negative_b_id in easy_negative_b_ids:
              training_triplets_ids.append({
                'a_id': a_id,
                'positive_b_id': positive_b_id,
                'negative_b_id': easy_negative_b_id,
                'type': 'easy'
              })

    print(f"Found {len(training_triplets_ids)} triplets.")

    # --- Step 3: Retrieve Text and Create InputExamples ---
    print("\n--- 3. Creating InputExample objects for training ---")
    train_examples = []

    #df1['combined_text'] = df1[text_columns_walmart].fillna('').apply(lambda row: ' '.join(row), axis=1)
    #df2['combined_text'] = df2[text_columns_amazon].fillna('').apply(lambda row: ' '.join(row), axis=1)
    #df1['combined_text'] = df1['combined_text'].str.lower()
    #df2['combined_text'] = df2['combined_text'].str.lower()
    a_id_to_text = pd.Series(df1.title.values, index=df1.id).to_dict()
    b_id_to_text = pd.Series(df2.title.values, index=df2.id).to_dict()

    sentences1 = []
    sentences2 = []
    labels = []
    for triplet in training_triplets_ids:
        anchor_text =   a_id_to_text.get(triplet['a_id'])
        positive_text = b_id_to_text.get(triplet['positive_b_id'])
        negative_text = b_id_to_text.get(triplet['negative_b_id'])

        if anchor_text and positive_text and negative_text:
            train_examples.append(InputExample(texts=[anchor_text, positive_text, negative_text]))  #this is for tripletloss

            #train_examples.append(InputExample(texts=[anchor_text, positive_text], label=1))
            #train_examples.append(InputExample(texts=[anchor_text, negative_text], label=0))

            sentences1.append(anchor_text)
            sentences2.append(positive_text)
            labels.append(1)

            # Negative pair
            sentences1.append(anchor_text)
            sentences2.append(negative_text)
            labels.append(0)

    print(f"Final number of training examples: {len(train_examples)}")
    print("\nExample of a training triplet:")

    for t  in range(100):
       print(train_examples[t].texts)

    # Load the pre-trained MiniLM model that we will fine-tune
    model_name = 'all-MiniLM-L6-v2'
    #model_name = "all-mpnet-base-v2"
    #model_name = "roberta-base-nli-stsb-mean-tokens"
    model = SentenceTransformer(model_name)

    # TripletLoss requires a dataloader that creates smart batches.
    # We'll use a batch size of 32, but you can adjust this based on your GPU memory.
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)

    # Define the TripletLoss function. This is the heart of the fine-tuning.
    train_loss = losses.TripletLoss(model=model)
    #train_loss = losses.SoftmaxLoss(
    #    model=model,
    #    sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
    #    num_labels=2  # binary classification (match / non-match)
    #)

    print("--- Model, Loss Function, and DataLoader are Ready ---")

    # --- 2. Fine-Tune the Model ---

    # Configure the training
    num_epochs = 1  # 1-4 epochs is usually sufficient for fine-tuning on this task.
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)  # 10% of training steps for warm-up
    output_model_path = './data/walmart_amazon-finetuned-model'  # The path where the new model will be saved

    evaluator = evaluation.BinaryClassificationEvaluator(
        sentences1=sentences1,
        sentences2=sentences2,
        labels=labels  # 1 = match, 0 = non-match
    )
    # The model's .fit() method orchestrates the entire training process
    print("\n--- Starting the Fine-Tuning Process ---")
    print(f"This will run for {num_epochs} epochs.")
    st = time.time()
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        evaluation_steps=1000,
        evaluator=evaluator,
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


def embed(input_filename, output_filename, text_columns, model):
    """
    Loads a table, generates dense embeddings and MinHash vectors,
    and saves the result to a new file.
    """
    if not os.path.exists(input_filename):
        print(f"Error: Input file '{input_filename}' not found. Skipping.")
        return

    print(f"\nProcessing '{input_filename}'...")

    # Load the table, reading all data as strings to be safe
    df = pd.read_csv(input_filename, sep=",", dtype=str)
    print(f"Number of rows:{len(df)}")

    # --- Step 1: Prepare text data ---
    # Fill any missing values in text columns with an empty string

    for col in text_columns:
        if col not in df.columns:
            df[col] = ''
        else:
            df[col].fillna('', inplace=True)

    # Combine relevant text columns for dense embedding
    print("Creating combined text for dense embeddings...")

    #df[text_columns].apply(lambda x: x.str.lower())
    for col in text_columns:
        df[col] = df[col].str.lower()

    #df['combined_text'] = df[text_columns].fillna('').sum(axis=1)
    df['combined_text'] = df[text_columns].fillna('').apply(lambda row: ' '.join(row), axis=1)

    print("Generating dense embeddings... (This may take a while)")
    sentences = df['combined_text'].tolist()
    embeddings = model.encode(sentences, show_progress_bar=True)

    # Add the embeddings to the DataFrame in a new column 'v'
    df['v'] = [emb.tolist() for emb in embeddings]

    for col in text_columns:
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


    '''
    df = pd.read_csv(f'{folder}/amazon_walmart.tsv', sep='\t', dtype=str)

    is_amazon = df['dataset'] == 'amazon'
    is_walmart = df['dataset'] == 'walmart'

    # 3. Use the masks to filter the original DataFrame into two new ones.
    df_amazon = df[is_amazon].copy()
    df_walmart = df[is_walmart].copy()

    # Optional: Drop the 'dataset' column if you no longer need it in the final files.
    df_amazon = df_amazon.drop(columns=['dataset'])
    df_walmart = df_walmart.drop(columns=['dataset'])


    # 4. Save each new DataFrame to its own CSV file.
    # We use index=False to prevent writing the row numbers to the file.
    amazon_filename = f'{folder}/amazon_products.csv'
    walmart_filename = f'{folder}/walmart_products.csv'

    df_amazon.to_csv(amazon_filename, index=False)
    df_walmart.to_csv(walmart_filename, index=False)

    exit()
    '''



    embedding_model = SentenceTransformer(model_name)

    embed(
        input_filename=f'{folder}/walmart_products.csv',
        output_filename=f'{folder}/walmart_products.pqt',
        text_columns=["brand",  "category",  "modelno","shortdescr", "longdescr","price", "title"],
        #text_columns=["brand", "category", "dimensions", "longdescr", "modelno", "orig_techdetails", "price",
        #              "shortdescr", "techdetails", "title"],
        model=embedding_model
    )
    embed(
        input_filename=f'{folder}/amazon_products.csv',
        output_filename=f'{folder}/amazon_products.pqt',
        text_columns=["brand", "category", "modelno", "shortdescr", "longdescr", "price", "title"],
        #text_columns=["brand", "category", "dimensions", "longdescr", "modelno", "orig_techdetails", "price",
        #              "shortdescr", "techdetails", "title"],
        model=embedding_model
    )


    #fine_tune(["brand", "category", "dimensions", "longdescr", "modelno", "orig_techdetails", "price",
    #                  "shortdescr", "techdetails", "title"], ["brand", "category", "dimensions", "longdescr", "modelno", "orig_techdetails", "price",
    #                  "shortdescr", "techdetails", "title"])
    fine_tune(["brand", "category",  "modelno", "shortdescr", "longdescr","price", "title"],
              ["brand",  "category",  "modelno", "shortdescr","longdescr", "price", "title"])

    model_path = f'{folder}/walmart_amazon-finetuned-model'
    embedding_model = SentenceTransformer(model_path)
    embed(
        input_filename=f'{folder}/walmart_products.csv',
        output_filename=f'{folder}/walmart_products_tuned.pqt',
        #text_columns=["brand", "category", "dimensions", "longdescr", "modelno", "orig_techdetails", "price",
        #             "shortdescr", "techdetails", "title"],
        text_columns=["brand", "category",   "modelno",  "shortdescr","longdescr","price", "title"],
        model=embedding_model
    )
    embed(
        input_filename=f'{folder}/amazon_products.csv',
        output_filename=f'{folder}/amazon_products_tuned.pqt',
        #text_columns=["brand", "category", "dimensions", "longdescr", "modelno", "orig_techdetails", "price",
        #              "shortdescr", "techdetails", "title"],
        text_columns=["brand",  "category", "modelno", "shortdescr","longdescr","price", "title"],
        model=embedding_model
    )

