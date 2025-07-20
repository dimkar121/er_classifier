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
        idDBLP = r["idDBLP"]
        idScholar = r["idScholar"]
        if idScholar in truthD:
            ids = truthD[idScholar]
            ids.append(idDBLP)
            a += 1
        else:
            truthD[idScholar] = [idDBLP]
            a += 1
    matches = a
    print("No of matches=", matches)

    # ====================================================================--

    from tensorflow import keras  # Or `import keras` depending on your setup

    loaded_model_path = f'./data/er_Scholar_DBLP2_{model_name}.keras'
    loaded_model = keras.models.load_model(loaded_model_path)


    print("Model loaded successfully!")
    batch_size = 10_000
    num_candidates = 5
    
    vectors_dblp = df22['v'].tolist()    
    dblp_embeddings = np.array(vectors_dblp).astype(np.float32)
    d = dblp_embeddings.shape[1]
    vectors_acm = df11['v'].tolist()
    scholar_embeddings = np.array(vectors_acm).astype(np.float32)
    scholar_ids = np.array(df11['id'].tolist())
    dblp_ids = np.array(df22['id'].tolist())


    names1 = {row['id']: row['title'] for index, row in df11.iterrows()}
    names2 = {row['id']: row['title'] for index, row in df22.iterrows()}

    index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 60
    index.hnsw.efSearch = 16
    index.add(scholar_embeddings)

    tp = 0
    fp = 0
    start_time = time.time()
    df_scholar_indexed = df11.set_index('id')
    df_dblp_indexed = df22.set_index('id')
    for i in range(0, len(vectors_dblp), batch_size):

        dblps = dblp_embeddings[i: i + batch_size]
        dblp_ids_in_batch = dblp_ids[i: i + batch_size]
        distances, candidate_indices = index.search(dblps, num_candidates)  # return scholars
        flat_candidate_ids = candidate_indices.flatten()

        candidate_scholar_embeddings = scholar_embeddings[flat_candidate_ids]
        repeated_dblp_embeddings = np.repeat(dblps, num_candidates, axis=0)
        repeated_dblp_ids = np.repeat(dblp_ids_in_batch, num_candidates)

        features_list = []
        scholar_ids_in_batch = scholar_ids[flat_candidate_ids]
        for scholarId_, dblpId_ in zip(scholar_ids_in_batch, repeated_dblp_ids):
            name1 = names1[scholarId_]
            name2 = names2[dblpId_]
            jw = jellyfish.jaro_winkler_similarity(str(name1), str(name2))

            features_list.append([jw])
        features_array = np.array(features_list, dtype='float32')


        combined_embeddings = np.concatenate([candidate_scholar_embeddings, repeated_dblp_embeddings], axis=1)

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

        for predicted_status, scholar_ind, dblpId in zip(predicted_statuses, candidate_indices.flatten(), repeated_dblp_ids):

          scholarId = scholar_ids[scholar_ind]
          title1 = df_scholar_indexed.loc[scholarId, 'title']
          description1 = df_scholar_indexed.loc[scholarId, 'authors']
          title2 = df_dblp_indexed.loc[dblpId, 'title']
          description2 = df_dblp_indexed.loc[dblpId, 'authors']
          


          if predicted_status == 1:
            scholarId = scholar_ids[scholar_ind]
            if scholarId in truthD.keys():
              idDblps = truthD[scholarId]
              for idDblp in idDblps:
                   if idDblp == dblpId:
                       tp += 1
                   else: 
                       fp += 1
            else:
               fp+=1               
 


    recall = round(tp / matches, 2)
    precision = round(tp / (tp + fp), 2)
    return recall, precision


if __name__ == '__main__':
     model = "mpnet_ft"
     df22 = pd.read_parquet(f"./data/DBLP2_{model}.pqt")
     df11 = pd.read_parquet(f"./data/Scholar_{model}.pqt")
     truth = pd.read_csv("./data/truth_Scholar_DBLP.csv", sep=",", encoding="unicode_escape", keep_default_na=False)

     recall, precision = run(df11, df22, truth, model, 0.2)
     print(recall, precision)



