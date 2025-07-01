import pandas as pd
import numpy as np
from lshashpy3 import LSHash
from tqdm import tqdm
import time
import utilities

if __name__ == '__main__':
    truth_file="./data/truth_Scholar_DBLP.csv"
    id1t = "idDBLP"
    id2t =  "idScholar"
    df11 = pd.read_parquet(f"./data/Scholar_embedded_mini.pqt")
    df22 = pd.read_parquet(f"./data/DBLP2_embedded_mini.pqt")
    #df11 = pd.read_parquet(f"./data/Scholar_e5.pqt")
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

    # Normalize vectors for cosine similarity (angular distance in LSH)
    a_embeddings /= np.linalg.norm(a_embeddings, axis=1, keepdims=True)
    b_embeddings /= np.linalg.norm(b_embeddings, axis=1, keepdims=True)
    embedding_dim = a_embeddings.shape[1]
    lsh = LSHash(hash_size=15,  num_hashtables=50, input_dim=embedding_dim)

    # Index each vector along with its original ID
    start_time = time.time()
    for i, vec in enumerate(a_embeddings):
        a_id = a_ids[i]
        lsh.index(vec, extra_data=a_id)
    end_time = time.time()
    print(f"Index built in {end_time - start_time:.2f} seconds.")
    tp = 0
    fp = 0
    k_neighbors = 5
    start_time = time.time()
    all_found_pairs=[]
    for i in tqdm(range(b_embeddings.shape[0]), desc="Evaluating queries"):
        b_id = b_ids[i]
        # Query the LSH index. It returns a list of tuples: ((vector, id), distance)
        response = lsh.query(b_embeddings[i], num_results=k_neighbors, distance_func="cosine")
        a_ids_ = [item[0][1] for item in response]
        b_ids_=np.repeat(b_id, len(a_ids_))
        # Extract just the IDs of the retrieved neighbors
        batch_pairs_df = pd.DataFrame({
          'id_A': a_ids_,
           #'title_A': titles1,
          'id_B': b_ids_,
          #'title_B': titles2
        })
        all_found_pairs.append(batch_pairs_df)
    end_time = time.time()
    final_results_df = pd.concat(all_found_pairs, ignore_index=True)
    print(f"Total pairs generated: {len(final_results_df)}")
    # Apply the function to create a new 'is_a_match' column
    final_results_df['is_a_match'] = final_results_df.apply(lambda row: utilities.check_if_match(row, truthD), axis=1 )
    tp = final_results_df['is_a_match'].sum()
    fp = (final_results_df['is_a_match'] == 0).sum()
    print(f"Matching queries resolved in {end_time - start_time:.2f} seconds.")
    print(f"tp={tp} fp={fp}  LSH recall={round(tp/matches, 2)} precision={round(tp/(tp+fp), 2)}")



