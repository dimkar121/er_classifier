import pandas as pd
import numpy as np
import faiss
import time
from scipy.spatial.distance import jaccard
import re
import jellyfish



if __name__ == '__main__':
    truth = pd.read_csv("./data/truth_imdb_dbpedia.csv", sep="|", encoding="unicode_escape", keep_default_na=False)
    truthD = dict()
    a = 0
    for i, r in truth.iterrows():
        idimdb = r["D1"]
        iddbpedia = r["D2"]
        if idimdb in truthD:
            ids = truthD[idimdb]
            ids.append(iddbpedia)
            a += 1
            print(idimdb, ids)
        else:
            truthD[idimdb] = [iddbpedia]
    matches = len(truthD.keys()) + a
    print("No of matches=", matches)

    # ====================================================================--

    from tensorflow import keras  # Or `import keras` depending on your setup

    loaded_model_path = 'data/er_imdb_dbpedia.keras'
    # Load the model
    loaded_model = keras.models.load_model(loaded_model_path)

    #loaded_model = xgb.XGBClassifier()
    #loaded_model.load_model("./data/abt_buy_xgb_model.json")


    print("Model loaded successfully!")
    # You can verify the model architecture
    #loaded_model.summary()
    batch_size = 10_000
    num_candidates = 5
    d = 384
    phi = 0.10520531820505105065205350
    df11 = pd.read_parquet(f"./data/imdb_tuned.pqt")
    df22 = pd.read_parquet(f"./data/dbpedia_tuned.pqt")
    df11['id'] = pd.to_numeric(df11['id'], errors='coerce')
    df22['id'] = pd.to_numeric(df22['id'], errors='coerce')
    minhash_titles1 = {row['id']: row['title_v'] for index, row in df11.iterrows()}
    minhash_titles2 = {row['id']: row['title_v'] for index, row in df22.iterrows()}
    minhash_actors1 = {row['id']: row['starring_v'] for index, row in df11.iterrows()}
    minhash_actors2 = {row['id']: row['actor name_v'] for index, row in df22.iterrows()}
    titles1 = {row['id']: row['title'] for index, row in df11.iterrows()}
    titles2 = {row['id']: row['title'] for index, row in df22.iterrows()}

    vectors_dbpedia = df22['v'].tolist()
    dbpedia_embeddings = np.array(vectors_dbpedia).astype(np.float32)
    vectors_imdb = df11['v'].tolist()
    imdb_embeddings = np.array(vectors_imdb).astype(np.float32)
    imdb_ids = np.array(df11['id'].tolist())
    dbpedia_ids = np.array(df22['id'].tolist())

    index = faiss.IndexHNSWFlat(d, 32)
    index.hnsw.efConstruction = 60
    index.hnsw.efSearch = 16
    index.add(dbpedia_embeddings)

    tp = 0
    fp = 0
    start_time = time.time()


    for i in range(0, len(vectors_imdb), batch_size):

        imdbs = imdb_embeddings[i: i + batch_size]
        imdb_ids_in_batch = imdb_ids[i: i + batch_size]
        distances, candidate_indices = index.search(imdbs, num_candidates)  # return scholars

        flat_candidate_ids = candidate_indices.flatten()
        candidate_dbpedia_embeddings = dbpedia_embeddings[flat_candidate_ids]
        repeated_imdb_embeddings = np.repeat(imdbs, num_candidates, axis=0)
        repeated_imdb_ids = np.repeat(imdb_ids_in_batch, num_candidates)


        features_list = []
        dbpedia_ids_in_batch = dbpedia_ids[flat_candidate_ids]
        for dbpediaId_, imdbId_ in zip(dbpedia_ids_in_batch, repeated_imdb_ids):
            minhash_title1 = minhash_titles1[imdbId_]
            minhash_title2 = minhash_titles2[dbpediaId_]
            minhash_actor1 = minhash_actors1[imdbId_]
            minhash_actor2 = minhash_actors2[dbpediaId_]
            j_distance1 = jaccard(minhash_title1, minhash_title2)
            j_distance2 = jaccard(minhash_actor1, minhash_actor2)
            j_similarity1 = 1 - j_distance1
            j_similarity2 = 1 - j_distance2
            title1 = titles1[imdbId_]
            title2 = titles2[dbpediaId_]
            jw = jellyfish.jaro_winkler_similarity(str(title1), str(title2))
            features_list.append([j_similarity1, j_similarity2, jw])
        features_array = np.array(features_list, dtype='float32')


        combined_embeddings = np.concatenate([candidate_dbpedia_embeddings, repeated_imdb_embeddings], axis=1)

        emb1_batch = combined_embeddings[:, :384]
        emb2_batch = combined_embeddings[:, 384:]



        numerator = np.einsum('ij,ij->i', emb1_batch, emb2_batch)
        denominator = np.linalg.norm(emb1_batch, axis=1) * np.linalg.norm(emb2_batch, axis=1)
        epsilon = 1e-7
        cosine_sim_scores = (numerator / (denominator + epsilon)).reshape(-1, 1)
        features_array = np.concatenate([features_array, cosine_sim_scores], axis=1)

        diff_vectors = emb1_batch - emb2_batch
        product_vectors = emb1_batch * emb2_batch
        interactions = np.concatenate([diff_vectors, product_vectors], axis=1)


        predictions = loaded_model.predict( [combined_embeddings, interactions, features_array], verbose=0)

        predicted_statuses = (predictions > phi).astype(int).flatten()

        for predicted_status, dbpedia_ind, imdbId in zip(predicted_statuses, candidate_indices.flatten(), repeated_imdb_ids):

          if predicted_status == 1:
            dbpediaId = imdb_ids[dbpedia_ind]
            if imdbId in truthD.keys():
                iddbpedia = truthD[imdbId]
                if iddbpedia == dbpediaId:
                       tp += 1
                else:
                       fp += 1


    end_time = time.time()
    print(f"{tp} {fp} recall={round(tp / matches, 2)} precision={round(tp / (tp + fp), 2)} total matching time={end_time-start_time} seconds.")
