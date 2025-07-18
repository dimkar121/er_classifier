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
        idAbt = r["idAbt"]
        idBuy = r["idBuy"]
        if idAbt in truthD:
            ids = truthD[idAbt]
            ids.append(idBuy)
            a += 1
        else:
            truthD[idAbt] = [idBuy]
    matches = len(truthD.keys()) + a
    print("No of matches=", matches)

    # ====================================================================--

    from tensorflow import keras  # Or `import keras` depending on your setup

    loaded_model_path = f'./data/er_abt_buy_{model_name}.keras'
    loaded_model = keras.models.load_model(loaded_model_path)


    print("Model loaded successfully!")
    batch_size = 10_000
    num_candidates = 5
    
    vectors_buy = df22['v'].tolist()    
    buy_embeddings = np.array(vectors_buy).astype(np.float32)
    d = buy_embeddings.shape[1]
    vectors_abt = df11['v'].tolist()
    abt_embeddings = np.array(vectors_abt).astype(np.float32)
    abt_ids = np.array(df11['id'].tolist())
    buy_ids = np.array(df22['id'].tolist())


    names1 = {row['id']: row['name'] for index, row in df11.iterrows()}
    names2 = {row['id']: row['name'] for index, row in df22.iterrows()}

    index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 60
    index.hnsw.efSearch = 16
    index.add(abt_embeddings)

    tp = 0
    fp = 0
    start_time = time.time()
    tp_ = 0
    fp_=0
    df_abt_indexed = df11.set_index('id')
    df_buy_indexed = df22.set_index('id')
    for i in range(0, len(vectors_buy), batch_size):

        buys = buy_embeddings[i: i + batch_size]
        buy_ids_in_batch = buy_ids[i: i + batch_size]
        distances, candidate_indices = index.search(buys, num_candidates)  # return scholars
        flat_candidate_ids = candidate_indices.flatten()

        candidate_abt_embeddings = abt_embeddings[flat_candidate_ids]
        repeated_buy_embeddings = np.repeat(buys, num_candidates, axis=0)
        repeated_buy_ids = np.repeat(buy_ids_in_batch, num_candidates)

        features_list = []
        abt_ids_in_batch = abt_ids[flat_candidate_ids]
        for abtId_, buyId_ in zip(abt_ids_in_batch, repeated_buy_ids):
            name1 = names1[abtId_]
            name2 = names2[buyId_]
            jw = jellyfish.jaro_winkler_similarity(str(name1), str(name2))

            features_list.append([jw])
        features_array = np.array(features_list, dtype='float32')


        combined_embeddings = np.concatenate([candidate_abt_embeddings, repeated_buy_embeddings], axis=1)

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

        for predicted_status, abt_ind, buyId in zip(predicted_statuses, candidate_indices.flatten(), repeated_buy_ids):

          abtId = abt_ids[abt_ind]
          title1 = df_abt_indexed.loc[abtId, 'name']
          description1 = df_abt_indexed.loc[abtId, 'description']
          title2 = df_buy_indexed.loc[buyId, 'name']
          description2 = df_buy_indexed.loc[buyId, 'description']
          


          if predicted_status == 1:
            abtId = abt_ids[abt_ind]
            tpFound = False
            if abtId in truthD.keys():
                idBuys = truthD[abtId]
                for idBuy in idBuys:
                   if idBuy == buyId:
                       tp += 1
                       tpFound=True
                if not tpFound:
                      fp += 1



    recall = round(tp / matches, 2)
    precision = round(tp / (tp + fp), 2)
    return recall, precision


if __name__ == '__main__':
     model = "mpnet_ft"
     df22 = pd.read_parquet(f"./data/Buy_{model}.pqt")
     df11 = pd.read_parquet(f"./data/Abt_{model}.pqt")
     truth = pd.read_csv("./data/truth_abt_buy.csv", sep=",", encoding="unicode_escape", keep_default_na=False)

     recall, precision = run(df11, df22, truth, model, 0.15)
     print(recall, precision)



