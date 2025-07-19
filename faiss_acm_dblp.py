import pandas as pd
import numpy as np
import faiss
import time
from scipy.spatial.distance import jaccard
import re
import jellyfish
import pickle



def run(df11, df22, truth, model_name, phi):
    truthD = dict()
    a = 0
    for i, r in truth.iterrows():
        idACM = r["idACM"]
        idDBLP = r["idDBLP"]
        if idACM in truthD:
            ids = truthD[idACM]
            ids.append(idDBLP)
            a += 1
        else:
            truthD[idACM] = [idDBLP]
    matches = len(truthD.keys()) + a
    print("No of matches=", matches)

    # ====================================================================--

    from tensorflow import keras  # Or `import keras` depending on your setup

    loaded_model_path = f'./data/er_ACM_DBLP_{model_name}.keras'
    loaded_model = keras.models.load_model(loaded_model_path)


    print("Model loaded successfully!")
    batch_size = 10_000
    num_candidates = 5
    
    vectors_dblp = df22['v'].tolist()    
    dblp_embeddings = np.array(vectors_dblp).astype(np.float32)
    d = dblp_embeddings.shape[1]
    vectors_acm = df11['v'].tolist()
    acm_embeddings = np.array(vectors_acm).astype(np.float32)
    acm_ids = np.array(df11['id'].tolist())
    dblp_ids = np.array(df22['id'].tolist())


    names1 = {row['id']: row['title'] for index, row in df11.iterrows()}
    names2 = {row['id']: row['title'] for index, row in df22.iterrows()}

    index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 60
    index.hnsw.efSearch = 16
    index.add(dblp_embeddings)

    tp = 0
    fp = 0
    start_time = time.time()
    df_acm_indexed = df11.set_index('id')
    df_dblp_indexed = df22.set_index('id')
    for i in range(0, len(vectors_acm), batch_size):

        acms = acm_embeddings[i: i + batch_size]
        acm_ids_in_batch = acm_ids[i: i + batch_size]
        distances, candidate_indices = index.search(acms, num_candidates)  # return scholars
        flat_candidate_ids = candidate_indices.flatten()

        candidate_dblp_embeddings = dblp_embeddings[flat_candidate_ids]
        repeated_acm_embeddings = np.repeat(acms, num_candidates, axis=0)
        repeated_acm_ids = np.repeat(acm_ids_in_batch, num_candidates)

        features_list = []
        dblp_ids_in_batch = dblp_ids[flat_candidate_ids]
        for dblpId_, acmId_ in zip(dblp_ids_in_batch, repeated_acm_ids):
            name1 = names1[acmId_]
            name2 = names2[dblpId_]
            jw = jellyfish.jaro_winkler_similarity(str(name1), str(name2))

            features_list.append([jw])
        features_array = np.array(features_list, dtype='float32')


        combined_embeddings = np.concatenate([candidate_dblp_embeddings, repeated_acm_embeddings], axis=1)

        emb1_batch = combined_embeddings[:, :d]
        emb2_batch = combined_embeddings[:, d:]
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

        for predicted_status, dblp_ind, acmId in zip(predicted_statuses, candidate_indices.flatten(), repeated_acm_ids):

          dblpId = dblp_ids[dblp_ind]
          title1 = df_acm_indexed.loc[acmId, 'title']
          description1 = df_acm_indexed.loc[acmId, 'authors']
          title2 = df_dblp_indexed.loc[dblpId, 'title']
          description2 = df_dblp_indexed.loc[dblpId, 'authors']
          


          if predicted_status == 1:
            dblpId = dblp_ids[dblp_ind]
            tpFound = False
            if acmId in truthD.keys():
                idDblps = truthD[acmId]
                for idDblp in idDblps:
                   if idDblp == dblpId:
                       tp += 1
                       tpFound=True
                if not tpFound:
                      fp += 1



    recall = round(tp / matches, 2)
    precision = round(tp / (tp + fp), 2)
    return recall, precision


if __name__ == '__main__':
     model = "mpnet_ft"
     df22 = pd.read_parquet(f"./data/DBLP_{model}.pqt")
     df11 = pd.read_parquet(f"./data/ACM_{model}.pqt")
     truth = pd.read_csv("./data/truth_ACM_DBLP.csv", sep=",", encoding="unicode_escape", keep_default_na=False)

     recall, precision = run(df11, df22, truth, model, 0.15)
     print(recall, precision)



