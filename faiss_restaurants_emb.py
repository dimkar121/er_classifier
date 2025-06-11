import pandas as pd
import numpy as np
import faiss
import time
from scipy.spatial.distance import jaccard
import re
import jellyfish



if __name__ == '__main__':
    truth = pd.read_csv("./data/truth_fodors_zagats.csv", sep=",", encoding="utf-8", keep_default_na=False)
    truthD = dict()
    a = 0
    for i, r in truth.iterrows():
        idFodors = r["idFodors"]
        idZagats = r["idZagats"]
        truthD[idFodors] = idZagats
    matches = len(truthD.keys())
    print("No of matches=", matches)
    # ====================================================================--


    from tensorflow import keras

    loaded_model_path = './data/er_general.keras'
    # Load the model
    loaded_model = keras.models.load_model(loaded_model_path)



    print("Model loaded successfully!")
    batch_size = 10_000
    num_candidates = 5
    d = 384
    phi = 0.15
    df11 = pd.read_parquet(f"./data/fodors_embedded_mini.pqt")
    df22 = pd.read_parquet(f"./data/zagats_embedded_mini.pqt")

    vectors_zagats = df22['v'].tolist()
    zagats_embeddings = np.array(vectors_zagats).astype(np.float32)
    vectors_fodors = df11['v'].tolist()
    fodors_embeddings = np.array(vectors_fodors).astype(np.float32)
    fodors_ids = np.array(df11['id'].tolist())
    zagats_ids = np.array(df22['id'].tolist())

    index = faiss.IndexHNSWFlat(d, 32)
    index.hnsw.efConstruction = 60
    index.hnsw.efSearch = 16
    index.add(fodors_embeddings)

    tp = 0
    fp = 0
    start_time = time.time()


    for i in range(0, len(vectors_zagats), batch_size):

        zagatss = zagats_embeddings[i: i + batch_size]
        zagats_ids_in_batch = zagats_ids[i: i + batch_size]
        distances, candidate_indices = index.search(zagatss, num_candidates)  # return scholars

        flat_candidate_ids = candidate_indices.flatten()

        candidate_fodors_embeddings = fodors_embeddings[flat_candidate_ids]
        repeated_zagats_embeddings = np.repeat(zagatss, num_candidates, axis=0)
        repeated_zagats_ids = np.repeat(zagats_ids_in_batch, num_candidates)

        combined_embeddings = np.concatenate([candidate_fodors_embeddings, repeated_zagats_embeddings], axis=1)


        emb1_batch = combined_embeddings[:, :384]
        emb2_batch = combined_embeddings[:, 384:]

        diff_vectors = emb1_batch - emb2_batch
        product_vectors = emb1_batch * emb2_batch
        interactions = np.concatenate([diff_vectors, product_vectors], axis=1)

        predictions = loaded_model.predict( [combined_embeddings, interactions], verbose=0)

        predicted_statuses = (predictions > phi).astype(int).flatten()

        for predicted_status, fodor_ind, zagatsId in zip(predicted_statuses, candidate_indices.flatten(), repeated_zagats_ids):

          if predicted_status == 1:
            fodorId = fodors_ids[fodor_ind]
            if fodorId in truthD.keys():
                idZagats = truthD[fodorId]
                if idZagats == zagatsId:
                       tp += 1
                else:
                       fp += 1


    end_time = time.time()
    print(f"recall={round(tp / matches, 2)} precision={round(tp / (tp + fp), 2)} total matching time={end_time-start_time} seconds.")
