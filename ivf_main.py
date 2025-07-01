import pandas as pd
import numpy as np
import faiss
import time
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

    batch_size = 10_000
    num_candidates = 5
    vectors_b = df22['v'].tolist()
    b_embeddings = np.array(vectors_b).astype(np.float32)
    d = b_embeddings.shape[1]
    vectors_a = df11['v'].tolist()
    a_embeddings = np.array(vectors_a).astype(np.float32)
    a_ids = np.array(df11['id'].tolist())
    b_ids = df22['id'].tolist()
    nlist = 6 * int(np.sqrt(len(a_embeddings)))
    quantizer = faiss.IndexFlatL2(d)
    m = 48  # 64 # for dimensionality = 1024 e5 model m=48 for Mini with 384 components
    # nbits: The number of bits per sub-quantizer code. 8 is standard.
    bits = 8
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    #cpu_index = faiss.IndexIVFFlat(quantizer, d, nlist)
    start_time = time.time()
    index.train(a_embeddings)
    index.add(a_embeddings)
    end_time = time.time()
    print(f"FAISS IVF Indexing Time: {end_time - start_time} seconds.")

    tp = 0
    fp = 0
    start_time = time.time()
    all_found_pairs = []
    for i in range(0, len(b_embeddings), batch_size):
        bs = b_embeddings[i: i + batch_size]
        b_ids_in_batch = b_ids[i: i + batch_size]
        distances, candidate_indices = index.search(bs, num_candidates)  # return scholars
        flat_candidate_ids = candidate_indices.flatten()
        candidate_a_embeddings = a_embeddings[flat_candidate_ids]
        candidate_a_ids = a_ids[flat_candidate_ids]
        repeated_b_embeddings = np.repeat(bs, num_candidates, axis=0)
        repeated_b_ids = np.repeat(b_ids_in_batch, num_candidates)
        batch_pairs_df = pd.DataFrame({
            'id_A': candidate_a_ids,
            'id_B': repeated_b_ids,
        })
        all_found_pairs.append(batch_pairs_df)
    final_results_df = pd.concat(all_found_pairs, ignore_index=True)
    print(f"Total pairs generated: {len(final_results_df)}")
    # Apply the function to create a new 'is_a_match' column
    final_results_df['is_a_match'] = final_results_df.apply(lambda row: utilities.check_if_match(row, truthD), axis=1)
    end_time = time.time()
    tp = final_results_df['is_a_match'].sum()
    fp = (final_results_df['is_a_match'] == 0).sum()
    print(f"IVF recall={round(tp / matches, 2)} precision={round(tp / (tp + fp), 2)} total matching time={end_time - start_time} seconds.")

