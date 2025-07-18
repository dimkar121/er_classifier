import pandas as pd
import numpy as np
import faiss
import time
import utilities
import os

def run(df11, df22, truth, id1t, id2t):
    
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
    nlist = 4 * int(np.sqrt(len(a_embeddings)))
    if len(a_embeddings) < 20_000:
       nlist =  int(np.sqrt(len(a_embeddings)))

    m = 48
    if d > 384:
       m = 64 
    bits = 8 
    gpu_id = -1
    if faiss.get_num_gpus() > 0:
      gpu_id = 0 # Use the first available GPU
      print(f"Found {faiss.get_num_gpus()} GPUs. Using GPU ID {gpu_id}.")
      # --- THE CORRECT METHOD for High-Performance GPU Indexing ---
      # We use IndexIVFPQ, which is designed for speed and large datasets.
      # 1. Create a "quantizer" index. This is a flat index used to partition the space.
      quantizer = faiss.IndexFlatL2(d)
      cpu_index = faiss.IndexIVFPQ(quantizer, d, nlist, m, bits)
      
      res = faiss.StandardGpuResources()
       
      if d > 384:
        print("Use Look up table")
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index, co)
      else:
        gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)
      
      index = gpu_index
    else:
       print("Warning: No GPU found. FAISS will run on the CPU.")
       # Fallback to a CPU index if no GPU is found
       quantizer = faiss.IndexFlatL2(d)
       cpu_index = faiss.IndexIVFPQ(quantizer, d, nlist, m, bits)
       index = cpu_index

    index.nprobe = 64
    #cpu_index = faiss.IndexIVFFlat(quantizer, d, nlist)
    start_time = time.time()
    index.train(a_embeddings)
    index.add(a_embeddings)
    end_time = time.time()
    index_time = round(end_time - start_time, 2)
    print(f"FAISS IVFPQ Indexing Time: {end_time - start_time} seconds.")

    index_filename = "./index_mem_print"
    if gpu_id == 0:
       print("Saving to CPU") 
       cpu_index_to_save = faiss.index_gpu_to_cpu(index)
       print("Writing index")        
       faiss.write_index(cpu_index_to_save, index_filename )       
    else:   
       faiss.write_index(index, index_filename)    
    # Get file size in bytes and convert to megabytes
    file_size_bytes = os.path.getsize(index_filename)
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"FAISS Index Memory Footprint: {file_size_mb:.2f} MB")
    # Clean up the file
    os.remove(index_filename)


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
    matching_time= round(end_time - start_time,2)
    tp = final_results_df['is_a_match'].sum()
    fp = (final_results_df['is_a_match'] == 0).sum()
    recall=round(tp / matches, 2)
    precision=round(tp / (tp + fp), 2)
    #print(f"HNSW tp={tp} fp={fp}  recall={round(tp / matches, 2)} precision={round(tp / (tp + fp), 2)} total matching time={end_time - start_time} seconds.")
    return tp, fp, recall, precision,index_time, matching_time, file_size_mb

