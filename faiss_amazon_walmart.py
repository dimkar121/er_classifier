import pandas as pd
import numpy as np
import faiss
import time
from scipy.spatial.distance import jaccard
import re
import jellyfish
from itertools import product
import pickle


def run(df11, df22, truth, model_name, phi):
    truthD = dict()
    a = 0
    for i, r in truth.iterrows():
        id_walmart = r["id2"] #id2
        id_amazon = r["id1"]  #id1
        if not id_walmart in df22['id'].values or not id_amazon in df11['id'].values:
                #print(f"Disregarding {id_walmart} or {id_amazon}")
                continue

        if id_amazon in truthD:
            ids = truthD[id_amazon]
            ids.append(id_walmart)
            a += 1
        else:
            truthD[id_amazon] = [id_walmart]
    matches = len(truthD.keys()) + a
    print("No of matches=", matches)

    # ====================================================================--

    from tensorflow import keras  # Or `import keras` depending on your setup

    #loaded_model_path = './data/er_walmart_amazon.keras'
    loaded_model_path = f'./data/er_amazon_walmart_{model_name}.keras'
    # Load the model
    loaded_model = keras.models.load_model(loaded_model_path)


    print("Model loaded successfully!")
    batch_size = 10_000
    num_candidates = 5

    df11['title_minhash'] = df11['title_bytes'].apply(pickle.loads)
    df22['title_minhash'] = df22['title_bytes'].apply(pickle.loads)
    minhash_titles1 = {row['id']: row['title_minhash'] for index, row in df11.iterrows()}
    minhash_titles2 = {row['id']: row['title_minhash'] for index, row in df22.iterrows()}
    titles1 = {row['id']: row['title'] for index, row in df11.iterrows()}
    titles2 = {row['id']: row['title'] for index, row in df22.iterrows()}

    vectors_amazon = df11['v'].tolist()   #df11
    amazon_embeddings = np.array(vectors_amazon).astype(np.float32)
    d = amazon_embeddings.shape[1]
    vectors_walmart = df22['v'].tolist()  #df22
    walmart_embeddings = np.array(vectors_walmart).astype(np.float32)
    amazon_ids = np.array(df11['id'].tolist())
    walmart_ids = np.array(df22['id'].tolist())

    index = faiss.IndexHNSWFlat(d, 32)
    index.hnsw.efConstruction = 60
    index.hnsw.efSearch = 16
    index.add(walmart_embeddings)

    tp = 0
    fp = 0
    start_time = time.time()

    df1_indexed = df11.set_index('id')
    df2_indexed = df22.set_index('id')

    for i in range(0, len(vectors_amazon), batch_size):

        amazons = amazon_embeddings[i: i + batch_size]
        amazon_ids_in_batch = amazon_ids[i: i + batch_size]
        distances, candidate_indices = index.search(amazons, num_candidates)  # return scholars

        flat_candidate_ids = candidate_indices.flatten()
        candidate_walmart_embeddings = walmart_embeddings[flat_candidate_ids]
        repeated_amazon_embeddings = np.repeat(amazons, num_candidates, axis=0)
        repeated_amazon_ids = np.repeat(amazon_ids_in_batch, num_candidates)


        features_list = []
        walmart_ids_in_batch = walmart_ids[flat_candidate_ids]
        for walmartId_, amazonId_ in zip(walmart_ids_in_batch, repeated_amazon_ids):

            minhash_title1 = minhash_titles1[amazonId_]
            minhash_title2 = minhash_titles2[walmartId_]
            j_distance1 = minhash_title1.jaccard(minhash_title2)
            j_similarity1 = 1 - j_distance1
            title1 = titles1[amazonId_]
            title2 = titles2[walmartId_]
            jw = jellyfish.jaro_winkler_similarity(str(title1), str(title2))
            features_list.append([ j_similarity1, jw])

        features_array = np.array(features_list, dtype='float32')


        combined_embeddings = np.concatenate([candidate_walmart_embeddings, repeated_amazon_embeddings], axis=1)

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

        for predicted_status, walmart_ind, amazonId in zip(predicted_statuses, candidate_indices.flatten(), repeated_amazon_ids):

          if predicted_status == 1:
             walmartId = walmart_ids[walmart_ind]
             tpFound = False
             if amazonId in truthD.keys():
                idWalmarts = truthD[amazonId]
                for idWalmart in idWalmarts:
                   if walmartId == idWalmart:
                       tp += 1
                       tpFound = True
                if not tpFound:
                       fp += 1
          #else:
          #    if walmartId in truthD.keys():
          #        idAmazons = truthD[walmartId]
          #        for idAmazon in idAmazons:
          #            if amazonId == idAmazon:
          #                print(walmartId, df1_indexed.loc[walmartId, ['title','modelno','category','brand']], "--\n",
          #                    amazonId, df2_indexed.loc[amazonId, ['title','modelno','category','brand']] )
          #                print("--------------------------------------------")
    end_time = time.time()
    recall = round(tp / matches, 2)
    precision = round(tp / (tp + fp), 2)
    return recall, precision


if __name__ == '__main__':
    model = "mpnet_ft"
    df22 = pd.read_parquet(f"./data/walmart_products_{model}.pqt")
    df11 = pd.read_parquet(f"./data/amazon_products_{model}.pqt")

    df11['id'] = pd.to_numeric(df11['id'], errors='coerce')
    df11.dropna(subset=['id'], inplace=True)
    df11['id'] = df11['id'].astype(int)
    df22['id'] = pd.to_numeric(df22['id'], errors='coerce')
    df22.dropna(subset=['id'], inplace=True)
    df22['id'] = df22['id'].astype(int)
    df11.reset_index(drop=True, inplace=True)
    df22.reset_index(drop=True, inplace=True)

    truth = pd.read_csv("./data/truth_amazon_walmart.tsv", sep="\t", encoding="unicode_escape", keep_default_na=False)
    truth['id1'] = truth['id1'].astype(int)
    truth['id2'] = truth['id2'].astype(int)

    recall, precision = run(df11, df22, truth, model, 0.15)
    print(recall, precision)


