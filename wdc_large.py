import pandas as pd
import numpy as np
import faiss
import time
import scann


if __name__ == '__main__':
    batch_size = 100_000
    num_candidates = 5
    df11 = pd.read_parquet(f"./data/wdc/large/tableA.pqt")
    df22 = pd.read_parquet(f"./data/wdc/large/tableB.pqt")
    df11['id'] = pd.to_numeric(df11['id'])
    df22['id'] = pd.to_numeric(df22['id'])
    a_id_to_title = pd.Series(df11.title.values, index=df11.id).to_dict()
    b_id_to_title = pd.Series(df22.title.values, index=df22.id).to_dict()
    vectors_b = df22['v'].tolist()
    b_embeddings = np.array(vectors_b).astype(np.float32)
    vectors_a = df11['v'].tolist()
    a_embeddings = np.array(vectors_a).astype(np.float32)
    d = a_embeddings.shape[1]
    a_ids = np.array(df11['id'].tolist())
    b_ids = np.array(df22['id'].tolist())
    if faiss.get_num_gpus() > 0:
        gpu_id = 0  # Use the first available GPU
        print(f"Found {faiss.get_num_gpus()} GPUs. Using GPU ID {gpu_id}.")
        # --- THE CORRECT METHOD for High-Performance GPU Indexing ---
        # We use IndexIVFPQ, which is designed for speed and large datasets.
        # 1. Create a "quantizer" index. This is a flat index used to partition the space.
        quantizer = faiss.IndexFlatL2(d)
        # 2. Define the main IVFPQ index on the CPU first.
        # nlist: The number of cells to partition the vector space into. A key tuning parameter.
        nlist = 1024
        # m: The number of sub-quantizers for Product Quantization.
        m = 8
        # nbits: The number of bits per sub-quantizer code. 8 is standard.
        bits = 8
        cpu_index = faiss.IndexIVFPQ(quantizer, d, nlist, m, bits)
        # 3. Create a GPU resources object and clone the CPU index to the GPU.
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)
        search_index = gpu_index
    else:
        print("Warning: No GPU found. FAISS will run on the CPU.")
        # Fallback to a CPU index if no GPU is found
        quantizer = faiss.IndexFlatL2(d)
        cpu_index = faiss.IndexIVFPQ(quantizer, d, 1024, 8, 8)
        search_index = cpu_index


    start_time = time.time()
    num_samples_to_get = 400_000
    random_indices = np.random.choice(
        b_embeddings.shape[0],
        size=num_samples_to_get,
        replace=False
    )
    vector_sample = b_embeddings[random_indices]
    search_index.train(vector_sample)
    search_index.nprobe = 30
    search_index.add(b_embeddings)
    end_time = time.time()
    print(f"FAISS IVFPQ Indexing Time: {end_time - start_time} seconds.")

    start_time = time.time()
    index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 60
    index.hnsw.efSearch = 16
    start_time = time.time()
    index.add(b_embeddings)
    end_time = time.time()
    print(f"FAISS HNSW Indexing Time: {end_time - start_time} seconds.")


    start_time = time.time()
    num_leaves = int(np.sqrt(len(b_embeddings)))
    # The number of leaves to search. A key trade-off for speed vs. accuracy.
    num_leaves_to_search = 50 
    # We use score_ah for Asymmetric Hashing, which is ScaNN's powerful quantization method.
    # The 2 means we use 2 bytes per dimension for compression.
    searcher = scann.scann_ops_pybind.builder(b_embeddings, 10, "dot_product") \
      .tree(num_leaves=num_leaves, num_leaves_to_search=num_leaves_to_search, training_sample_size=25000) \
      .score_ah(2, anisotropic_quantization_threshold=0.2) \
      .reorder(100) \
      .build()
    print(f"ScaNN index built successfully with {num_leaves} leaves.")
    end_time = time.time()
    print(f"Scann Indexing Time: {end_time - start_time} seconds.")
    



    tp = 0
    fp = 0
    start_time = time.time()
    all_found_pairs = []
    for i in range(0, len(a_embeddings), batch_size):
        bs = a_embeddings[i: i + batch_size]
        a_ids_in_batch = a_ids[i: i + batch_size]
        candidate_indices, distances = searcher.search_batched(bs, final_num_neighbors=num_candidates)       
        #distances, candidate_indices = search_index.search(bs, num_candidates)  # return scholars
        flat_candidate_ids = candidate_indices.flatten()
        candidate_b_embeddings = b_embeddings[flat_candidate_ids]
        repeated_a_embeddings = np.repeat(bs, num_candidates, axis=0)
        repeated_a_ids = np.repeat(a_ids_in_batch, num_candidates)
        candidate_b_ids = b_ids[flat_candidate_ids]
        titles1 = pd.Series(repeated_a_ids).map(a_id_to_title).values
        titles2 = pd.Series(candidate_b_ids).map(b_id_to_title).values
        batch_pairs_df = pd.DataFrame({
            'id_A': repeated_a_ids,
            'title_A': titles1,
            'id_B': candidate_b_ids,
            'title_B': titles2
        })
        all_found_pairs.append(batch_pairs_df)


    end_time = time.time()
    print(f"Scann Matching Time: {end_time - start_time} seconds.")

    
    tp = 0
    fp = 0
    start_time = time.time()
    all_found_pairs = []
    for i in range(0, len(a_embeddings), batch_size):
         bs = a_embeddings[i: i + batch_size]
         a_ids_in_batch = a_ids[i: i + batch_size]
         distances, candidate_indices = search_index.search(bs, num_candidates)  # return scholars
         flat_candidate_ids = candidate_indices.flatten()
         candidate_b_embeddings = b_embeddings[flat_candidate_ids]
         repeated_a_embeddings = np.repeat(bs, num_candidates, axis=0)
         repeated_a_ids = np.repeat(a_ids_in_batch, num_candidates)
         candidate_b_ids = b_ids[flat_candidate_ids]
         titles1 = pd.Series(repeated_a_ids).map(a_id_to_title).values
         titles2 = pd.Series(candidate_b_ids).map(b_id_to_title).values
         batch_pairs_df = pd.DataFrame({
             'id_A': repeated_a_ids,
             'title_A': titles1,
             'id_B': candidate_b_ids,
             'title_B': titles2
         })
         all_found_pairs.append(batch_pairs_df)


    end_time = time.time()
    print(f"FAISS IVFPQ Matching Time: {end_time - start_time} seconds.")


    tp = 0
    fp = 0
    start_time = time.time()
    all_found_pairs = []
    for i in range(0, len(a_embeddings), batch_size):
          bs = a_embeddings[i: i + batch_size]
          a_ids_in_batch = a_ids[i: i + batch_size]
          distances, candidate_indices = index.search(bs, num_candidates)  # return scholars
          flat_candidate_ids = candidate_indices.flatten()
          candidate_b_embeddings = b_embeddings[flat_candidate_ids]
          repeated_a_embeddings = np.repeat(bs, num_candidates, axis=0)
          repeated_a_ids = np.repeat(a_ids_in_batch, num_candidates)
          candidate_b_ids = b_ids[flat_candidate_ids]
          titles1 = pd.Series(repeated_a_ids).map(a_id_to_title).values
          titles2 = pd.Series(candidate_b_ids).map(b_id_to_title).values
          batch_pairs_df = pd.DataFrame({
              'id_A': repeated_a_ids,
              'title_A': titles1,
              'id_B': candidate_b_ids,
              'title_B': titles2
          })
          all_found_pairs.append(batch_pairs_df)


    end_time = time.time()
    print(f"FAISS HNSW Matching Time: {end_time - start_time} seconds.")


    exit()
    final_results_df = pd.concat(all_found_pairs, ignore_index=True)
    print(f"Total pairs generated: {len(final_results_df)}")
    df_subset = final_results_df.head(100)
    print("--- Displaying the first 100 title pairs ---")
    # .iterrows() yields both the index and the row (as a Series)
    for index, row in df_subset.iterrows():
      # Access the data from the 'title_A' and 'title_B' columns for the current row
      title1 = row['title_A']
      title2 = row['title_B']
      print(index, title1, title2)



