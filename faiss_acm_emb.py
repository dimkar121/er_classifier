import pandas as pd
import numpy as np
import faiss
import time

if __name__ == '__main__':
    truth = pd.read_csv("./data/truth_ACM_DBLP.csv", sep=",", encoding="utf-8", keep_default_na=False)
    truthD = dict()
    a = 0

    for i, r in truth.iterrows():
        idDBLP = r["idDBLP"]
        idACM = r["idACM"]
        if idACM in truthD:
            ids = truthD[idACM]
            ids.append(idDBLP)
            a += 1
        else:
            truthD[idACM] = [idDBLP]
            a += 1
    matches = a
    print("No of matches=", matches)

    # ====================================================================--

    from tensorflow import keras  # Or `import keras` depending on your setup

    loaded_model_path = './data/er_general.keras'
    # Load the model
    loaded_model = keras.models.load_model(loaded_model_path)
    print("Model loaded successfully!")
    loaded_model.summary()
    batch_size = 5_000
    num_candidates = 5
    d = 384
    phi = 0.19
    df11 = pd.read_parquet(f"./data/ACM_embedded_mini.pqt")
    df22 = pd.read_parquet(f"./data/DBLP_embedded_mini.pqt")
    vectors_dblp = df22['v'].tolist()
    dblp_embeddings = np.array(vectors_dblp).astype(np.float32)
    vectors_acm = df11['v'].tolist()
    acm_embeddings = np.array(vectors_acm).astype(np.float32)
    acm_ids = np.array(df11['id'].tolist())
    dblp_ids = df22['id'].tolist()



    index = faiss.IndexHNSWFlat(d, 32)
    index.hnsw.efConstruction = 60
    index.hnsw.efSearch = 16
    index.add(dblp_embeddings)




    tp = 0
    fp = 0
    start_time = time.time()


    for i in range(0, len(vectors_acm), batch_size):

        acms = acm_embeddings[i: i + batch_size]
        acm_ids_in_batch = acm_ids[i: i + batch_size]
        distances, candidate_indices = index.search(acms, num_candidates)  # return scholars
        flat_candidate_ids = candidate_indices.flatten()
        candidate_dblp_embeddings = dblp_embeddings[flat_candidate_ids]
        repeated_acm_embeddings = np.repeat(acms, num_candidates, axis=0)
        repeated_acm_ids = np.repeat(acm_ids_in_batch, num_candidates)

        combined_embeddings = np.concatenate([repeated_acm_embeddings, candidate_dblp_embeddings], axis=1)

        emb1_batch = combined_embeddings[:, :384]
        emb2_batch = combined_embeddings[:, 384:]

        diff_vectors = emb1_batch - emb2_batch
        product_vectors = emb1_batch * emb2_batch
        interactions = np.concatenate([diff_vectors, product_vectors], axis=1)
        predictions = loaded_model.predict([combined_embeddings, interactions], verbose=0)
        predicted_statuses = (predictions > phi).astype(int).flatten()

        for predicted_status, dblp_ind, acmId in zip(predicted_statuses, candidate_indices.flatten(), repeated_acm_ids):

          if predicted_status == 1:
            if acmId in truthD.keys():
                tpFound = False
                idDBLPs = truthD[acmId]
                for idDBLP in idDBLPs:
                    if idDBLP == dblp_ids[dblp_ind]:
                        tp += 1
                        tpFound = True
                if tpFound == False:
                    fp += 1


    end_time = time.time()
    print(f"recall={round(tp / matches, 2)} precision={round(tp / (tp + fp), 2)} total matching time={end_time-start_time} seconds.")
