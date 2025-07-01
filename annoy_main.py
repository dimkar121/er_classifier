import pandas as pd
import numpy as np
from annoy import AnnoyIndex
import time
from tqdm import tqdm
import utilities



if __name__ == '__main__':
    truth_file="./data/truth_Scholar_DBLP.csv"
    id1t = "idDBLP"
    id2t =  "idScholar"
    df11 = pd.read_parquet(f"./data/Scholar_embedded_mini.pqt")
    df22 = pd.read_parquet(f"./data/DBLP2_embedded_mini.pqt")
    #df11 = pd.read_parquet(f"/data/Scholar_e5.pqt")
    #df22 = pd.read_parquet(f"./data/DBLP2_e5.pqt")

    truth = pd.read_csv(truth_file, sep=",", encoding="utf-8", keep_default_na=False)
    truthD = dict()
    a = 0

    for i, r in truth.iterrows():
        id1 = r[id1t]
        id2 = r[id2t]
        if id2 in truthD:
            ids = truthD[id2]
            ids.append(id1)
            a += 1
        else:
            truthD[id2] = [id1]
            a += 1
    matches = a
    print("No of matches=", matches)

    # ====================================================================--

    vectors_b = df22['v'].tolist()
    b_embeddings = np.array(vectors_b).astype(np.float32)
    d = b_embeddings.shape[1]
    vectors_a = df11['v'].tolist()
    a_embeddings = np.array(vectors_a).astype(np.float32)
    a_ids = np.array(df11['id'].tolist())
    b_ids = df22['id'].tolist()
    #b_id_to_title = pd.Series(df22.title.values, index=df22.id).to_dict()
    #a_id_to_title = pd.Series(df11.title.values, index=df11.id).to_dict()

    # Normalize vectors for cosine similarity (angular distance in Annoy)
    a_embeddings /= np.linalg.norm(a_embeddings, axis=1, keepdims=True)
    b_embeddings /= np.linalg.norm(b_embeddings, axis=1, keepdims=True)
    embedding_dim = a_embeddings.shape[1]
    # --- 2. BUILD THE ANNOY INDEX ---
    print("\n--- 2. Building the Annoy index ---")

    # Initialize the index.
    # f: The dimension of the vectors.
    # metric: Can be 'angular' (for cosine), 'euclidean', 'manhattan', etc.
    # 'angular' is the correct choice for normalized semantic embeddings.
    annoy_index = AnnoyIndex(embedding_dim, 'angular')

    # Add items to the index one by one. The first argument is the item's integer index.
    print("Adding items to the index...")
    for i in tqdm(range(a_embeddings.shape[0])):
         annoy_index.add_item(i, a_embeddings[i])

    # Build the index. The argument is the number of trees to build.
    # More trees gives higher precision but takes longer to build. 10-100 is a common range.
    print("\nBuilding the forest of trees...")
    start_time = time.time()
    n_trees=150
    annoy_index.build(n_trees)
    end_time = time.time()
    print(f"Index built in {end_time - start_time:.2f} seconds.")

    # Optionally, save the index to disk for later use
    #annoy_index.save('my_annoy_index.ann')
    #print("Index saved to disk.")

    # --- 3. SEARCH THE ANNOY INDEX ---
    # If you were in a different script, you would first load the index:
    # loaded_index = AnnoyIndex(embedding_dim, 'angular')
    # loaded_index.load('my_annoy_index.ann')

    print("\n--- 3. Searching the index ---")
    k_neighbors = 5 # The number of nearest neighbors to find

    all_neighbors = []
    all_distances = []
    tp=0
    fp=0
    total_candidates_generated = 0
    total_positives_in_gold_standard = sum(len(v) for v in truthD.values())
    all_found_pairs = []
    start_time = time.time()
    # Query the index for each vector
    for i in range(b_embeddings.shape[0]):
      # .get_nns_by_vector returns the indices and distances of the nearest neighbors
      neighbors, distances = annoy_index.get_nns_by_vector(
        b_embeddings[i],
        k_neighbors,
        search_k=n_trees * k_neighbors,
        include_distances=True
      )
      total_candidates_generated += len(neighbors)
      b_id = b_ids[i]
      #if dblp_id not in truthD:
       #      continue
      a_ids_ = []
      b_ids_=np.repeat(b_id, len(neighbors))
      for neighbor_idx in neighbors:
         a_id = a_ids[neighbor_idx]
         a_ids_.append(a_id)
      batch_pairs_df = pd.DataFrame({
          'id_A': a_ids_,
          'id_B': b_ids_,
      })
      all_found_pairs.append(batch_pairs_df)

    final_results_df = pd.concat(all_found_pairs, ignore_index=True)
    print(f"Total pairs generated: {len(final_results_df)}")
    end_time = time.time()
    # Apply the function to create a new 'is_a_match' column
    final_results_df['is_a_match'] = final_results_df.apply(lambda row: utilities.check_if_match(row, truthD), axis=1 )
    tp = final_results_df['is_a_match'].sum()
    fp = (final_results_df['is_a_match'] == 0).sum()
    print(f"Matching queries resolved in {end_time - start_time:.2f} seconds.")
    print(f"tp={tp} fp={fp} total_candidates={total_candidates_generated} Annoy recall={round(tp/matches, 2)} precision={round(tp/(tp+fp), 2)}")

