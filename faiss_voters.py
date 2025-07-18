import pandas as pd
import numpy as np
import faiss
import time

def run(df11, df22, truth, model_name, phi):
    truthD = dict()
    a = 0

    for i, r in truth.iterrows():
        id1 = r["id1"]
        id2 = r["id2"]
        if id1 in truthD:
            ids = truthD[id1]
            if id2 in ids:
                print(f" {id2} already exists for {id1}")
            ids.append(id2)
            a += 1
        else:
            if id1 != id2:
                print(f" different ids {id1} and {id2}")
            truthD[id1] = [id2]
            a += 1
    matches = a
    print("No of matches=", matches)

    # ====================================================================--

    from tensorflow import keras  # Or `import keras` depending on your setup

    loaded_model_path = f'./data/er_voters_{model_name}.keras'
    print("Model: {loaded_model_path}")
    # Load the model
    loaded_model = keras.models.load_model(loaded_model_path)
    print("Model loaded successfully!")
    loaded_model.summary()
    batch_size = 10_000
    num_candidates = 5
    phi = 0.1620151
    vectors_votersb = df22['v'].tolist()
    votersb_embeddings = np.array(vectors_votersb).astype(np.float32)
    d = votersb_embeddings.shape[1]
    vectors_votersa = df11['v'].tolist()
    votersa_embeddings = np.array(vectors_votersa).astype(np.float32)
    votersa_ids = np.array(df11['id'].tolist())
    votersb_ids = df22['id'].tolist()



    index = faiss.IndexHNSWFlat(d, 32)
    index.hnsw.efConstruction = 60
    index.hnsw.efSearch = 16
    index.add(votersb_embeddings)


    tp = 0
    fp = 0
    start_time = time.time()

    tps = set()
    for i in range(0, len(vectors_votersb), batch_size):

        votersbs = votersb_embeddings[i: i + batch_size]
        votersb_ids_in_batch = votersb_ids[i: i + batch_size]
        distances, candidate_indices = index.search(votersbs, num_candidates)  # return scholars
        flat_candidate_ids = candidate_indices.flatten()
        candidate_votersa_embeddings = votersa_embeddings[flat_candidate_ids]
        repeated_votersb_embeddings = np.repeat(votersbs, num_candidates, axis=0)
        repeated_votersb_ids = np.repeat(votersb_ids_in_batch, num_candidates)
        combined_embeddings = np.concatenate([repeated_votersb_embeddings, candidate_votersa_embeddings], axis=1)

        emb1_batch = combined_embeddings[:, :d]
        emb2_batch = combined_embeddings[:, d:]
        numerator = np.einsum('ij,ij->i', emb1_batch, emb2_batch)
        denominator = np.linalg.norm(emb1_batch, axis=1) * np.linalg.norm(emb2_batch, axis=1)
        epsilon = 1e-7
        cosine_sim_scores = (numerator / (denominator + epsilon)).reshape(-1, 1)
        features =  cosine_sim_scores


        diff_vectors = emb1_batch - emb2_batch
        product_vectors = emb1_batch * emb2_batch
        interactions = np.concatenate([diff_vectors, product_vectors], axis=1)
        predictions = loaded_model.predict([combined_embeddings, interactions, features], verbose=0)

        predicted_statuses = (predictions > phi).astype(int).flatten()

        for predicted_status, votersa_ind, votersbId in zip(predicted_statuses, candidate_indices.flatten(), repeated_votersb_ids):

          if predicted_status == 1:
            votersaId = votersa_ids[votersa_ind]
            if votersaId in truthD.keys():
                tpFound = False
                idvotersb = truthD[votersaId]
                #print(idvotersb, votersb_ids[votersb_ind])
                if votersaId == idvotersb:
                    if not votersaId in tps:
                        tp += 1
                        tps.add(votersaId)
                else:
                       fp += 1


    recall=round(tp / matches, 2)
    precision=round(tp / (tp + fp), 2)
    print(f" recall={recall} precision={precision} ")
    return recall, precision


