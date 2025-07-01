import pandas as pd
import numpy as np
import faiss
import time
import pyarrow.parquet as pq


if __name__ == '__main__':
    batch_size = 100_000
    num_candidates = 10
    df11 = pd.read_parquet(f"./data/wdc/large/tableA.pqt")
    #df22 = pd.read_parquet(f"/content/drive/MyDrive/data/large/tableB.pqt")
    chunk_size = 200000 # How many rows to process at a time. Adjust based on your RAM.
    sample_size_for_training = 200000 # How many vectors to use for training the index.
    # Open the Parquet file without loading it all into memory
    parquet_file = pq.ParquetFile(f"./data/wdc/large/tableB.pqt")
    # Load a sample of the data for training
    print(f"Loading a sample of {sample_size_for_training} rows for training...")
    sample_iter = parquet_file.iter_batches(batch_size=sample_size_for_training)
    first_batch = next(sample_iter)
    sample_df = first_batch.to_pandas()

    #df11['id'] = pd.to_numeric(df11['id'])
    #df22['id'] = pd.to_numeric(df22['id'])
    a_id_to_title = pd.Series(df11.title.values, index=df11.id).to_dict()
    vectors_a = df11['v'].tolist()
    a_embeddings = np.array(vectors_a).astype(np.float32)
    d = a_embeddings.shape[1]
    a_ids = np.array(df11['id'].tolist())
    if faiss.get_num_gpus() > 0:
      gpu_id = 0 # Use the first available GPU
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
       nlist=1024
       cpu_index = faiss.IndexIVFPQ(quantizer, d, nlist, 8, 8)
       search_index = cpu_index

    vectors_sample = sample_df['v'].tolist()
    vectors_sample_embeddings = np.array(vectors_sample).astype(np.float32)
    search_index.train(vectors_sample_embeddings)

    print(f"Index trained with {search_index.nlist} clusters (inverted lists).")
    print("-" * 50)


    search_index.nprobe = 30
    all_embeddings_list = []
    all_ids_list = []
    id_to_title_dict = {}
    # We re-iterate from the beginning of the file
    for i, batch in enumerate(parquet_file.iter_batches(batch_size=chunk_size, columns=['id','v', 'title'])):
       print(f"Processing chunk {i+1}...")    
       # Convert the current chunk to a pandas DataFrame
       chunk_df = batch.to_pandas()    
       vectors_chunk = chunk_df['v'].tolist()
       b_embeddings_chunk = np.array(vectors_chunk).astype(np.float32)    
       # Generate embeddings for the current chunk
       # Add the chunk of vectors to the trained index
       search_index.add(b_embeddings_chunk)    
       all_embeddings_list.append(vectors_chunk)
       all_ids_list.extend(chunk_df['id'].tolist())
       id_to_title_dict.update(pd.Series(chunk_df.title.values, index=chunk_df.id).to_dict())
       print(f"Added {len(vectors_chunk)} vectors. Total vectors in index: {search_index.ntotal}")

    invlists = faiss.extract_index_ivf(search_index).invlists
    print(invlists)
    # Now you can inspect the clusters
    # For example, to see which vector IDs are in cluster #5:
    cluster_id_to_check = 5
    ids_in_cluster = faiss.rev_swig_ptr(invlists.get_ids(cluster_id_to_check), invlists.list_size(cluster_id_to_check))
    print(f"Vector IDs in cluster {cluster_id_to_check}:")
    print(ids_in_cluster)
    exit()



    
    try:
        # Concatenate all embedding chunks into one large NumPy array
        b_embeddings = np.vstack(all_embeddings_list)
        print(f"Successfully created 'b_embeddings' array with shape: {b_embeddings.shape}")

        # The dictionary is already created
        b_id_to_title = id_to_title_dict
        print(f"Successfully created 'b_id_to_title' dictionary with {len(b_id_to_title)} entries.")

        # Create the final ID array
        b_ids = np.array(all_ids_list)
        print(f"Successfully created 'b_ids' array with shape: {b_ids.shape}")

    except MemoryError:
        print("\nFATAL ERROR: A MemoryError occurred during final assembly.")
        print("The combined size of the embeddings is too large to fit into your available RAM.")
        print("Consider increasing your system's RAM or using a machine with more memory.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during final assembly: {e}")

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
    print(f"Time: {end_time - start_time} seconds.")
    final_results_df = pd.concat(all_found_pairs, ignore_index=True)
    print(f"Total pairs generated: {len(final_results_df)}")
    print("\nFirst 10 generated pairs:")
    print(final_results_df.head(10))

