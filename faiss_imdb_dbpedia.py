import pandas as pd
import numpy as np
import faiss
import time
from scipy.spatial.distance import jaccard
import re
import jellyfish
import utilities
import pickle

def run(df11, df22, truth, model_name, phi):
    truthD = dict()
    a = 0
    for i, r in truth.iterrows():
        idimdb = r["D1"]
        iddbpedia = r["D2"]
        if not idimdb in df11['id'].values or not iddbpedia in df22['id'].values:
            continue
        if idimdb in truthD:
            ids = truthD[idimdb]
            ids.append(iddbpedia)
            a += 1
            print(idimdb, ids)
        else:
            truthD[idimdb] = [iddbpedia]
    matches = len(truthD.keys()) + a
    print("No of matches=", matches, "phi=", phi, "model_name", model_name)

    # ====================================================================--

    from tensorflow import keras  # Or `import keras` depending on your setup

    loaded_model_path = f'./data/er_imdb_dbpedia_{model_name}.keras'
    loaded_model = keras.models.load_model(loaded_model_path)


    print("Model loaded successfully!")
    #loaded_model.summary()
    batch_size = 10_000
    num_candidates = 5
    
    #phi = 0.410520531820505105065205350

    #minhash_titles1 = {row['id']: row['title_v'] for index, row in df11.iterrows()}
    #minhash_titles2 = {row['id']: row['title_v'] for index, row in df22.iterrows()}
    #minhash_actors1 = {row['id']: row['starring_v'] for index, row in df11.iterrows()}
    #minhash_actors2 = {row['id']: row['actor name_v'] for index, row in df22.iterrows()}

    df11['title_minhash'] = df11['title_minhash_bytes'].apply(pickle.loads)
    df11['starring_minhash'] = df11['starring_minhash_bytes'].apply(pickle.loads)
    df22['title_minhash'] = df22['title_minhash_bytes'].apply(pickle.loads)
    df22['starring_minhash'] = df22['starring_minhash_bytes'].apply(pickle.loads)

   
    #df11['title_minhash'] = df11['title'].apply(utilities.get_minhash)
    #df22['title_minhash'] = df22['title'].apply(utilities.get_minhash)
    #df11['starring_minhash'] = df11['starring'].apply(utilities.get_minhash)
    #df22['starring_minhash'] = df22['actor name'].apply(utilities.get_minhash)
    minhash_titles1 = {row['id']: row['title_minhash'] for index, row in df11.iterrows()}
    minhash_titles2 = {row['id']: row['title_minhash'] for index, row in df22.iterrows()}
    minhash_actors1 = {row['id']: row['starring_minhash'] for index, row in df11.iterrows()}
    minhash_actors2 = {row['id']: row['starring_minhash'] for index, row in df22.iterrows()}
    titles1 = {row['id']: row['title'] for index, row in df11.iterrows()}
    titles2 = {row['id']: row['title'] for index, row in df22.iterrows()}

    vectors_dbpedia = df22['v'].tolist()
    dbpedia_embeddings = np.array(vectors_dbpedia).astype(np.float32)
    d = dbpedia_embeddings.shape[1]
    vectors_imdb = df11['v'].tolist()
    imdb_embeddings = np.array(vectors_imdb).astype(np.float32)
    imdb_ids = np.array(df11['id'].tolist())
    dbpedia_ids = np.array(df22['id'].tolist())
    index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 60
    index.hnsw.efSearch = 16
    index.add(imdb_embeddings)

    tp = 0
    fp = 0
    start_time = time.time()

    df1_indexed = df11.set_index('id')
    df2_indexed = df22.set_index('id')

    aa = 0
    for i in range(0, len(vectors_dbpedia), batch_size):

        dbpedias = dbpedia_embeddings[i: i + batch_size]
        dbpedia_ids_in_batch = dbpedia_ids[i: i + batch_size]
        distances, candidate_indices = index.search(dbpedias, num_candidates)  # return scholars

        flat_candidate_ids = candidate_indices.flatten()
        candidate_imdb_embeddings = imdb_embeddings[flat_candidate_ids]
        repeated_dbpedia_embeddings = np.repeat(dbpedias, num_candidates, axis=0)
        repeated_dbpedia_ids = np.repeat(dbpedia_ids_in_batch, num_candidates)


        features_list = []
        imdb_ids_in_batch = imdb_ids[flat_candidate_ids]
        for imdbId_, dbpediaId_ in zip(imdb_ids_in_batch, repeated_dbpedia_ids):
            #minhash_title1 = minhash_titles1[imdbId_]
            #minhash_title2 = minhash_titles2[dbpediaId_]
            #minhash_actor1 = minhash_actors1[imdbId_]
            #minhash_actor2 = minhash_actors2[dbpediaId_]
            #j_distance1 = jaccard(minhash_title1, minhash_title2)
            #j_distance2 = jaccard(minhash_actor1, minhash_actor2)
            #j_similarity1 = 1 - j_distance1
            #j_similarity2 = 1 - j_distance2
      
            minhash_title1 = minhash_titles1[imdbId_]
            minhash_title2 = minhash_titles2[dbpediaId_]
            minhash_actor1 = minhash_actors1[imdbId_]
            minhash_actor2 = minhash_actors2[dbpediaId_]
            j_distance1 = minhash_title1.jaccard(minhash_title2)
            j_distance2 = minhash_actor1.jaccard(minhash_actor2)
            j_similarity1 = 1 - j_distance1
            j_similarity2 = 1 - j_distance2
            title1 = titles1[imdbId_]
            title2 = titles2[dbpediaId_]
            jw = jellyfish.jaro_winkler_similarity(str(title1), str(title2))
            features_list.append([ j_distance1, j_distance2, j_similarity1, j_similarity2, jw])
        features_array = np.array(features_list, dtype='float32')


        combined_embeddings = np.concatenate([candidate_imdb_embeddings, repeated_dbpedia_embeddings], axis=1)

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

        for predicted_status, imdb_ind, dbpediaId in zip(predicted_statuses, candidate_indices.flatten(), repeated_dbpedia_ids):
          aa+=1   
          if predicted_status == 1:             
             imdbId = imdb_ids[imdb_ind]
             if imdbId in truthD.keys():
                iddbpedia = truthD[imdbId]
                if iddbpedia == dbpediaId:
                       tp += 1
                else:
                       fp += 1
             
             '''
             else:
               imdbId = imdb_ids[imdb_ind]
               if imdbId in truthD.keys():
                 iddbpedia = truthD[imdbId]
                 if iddbpedia == dbpediaId:
                        tp += 1
                 else:
                        fp += 1
              ''' 
 
             #print(df1_indexed.loc[imdbId, 'title'], "--", df2_indexed.loc[dbpediaId, 'title'] )
    end_time = time.time()     
    recall=round(tp / matches, 2) 
    precision=round(tp / (tp + fp), 2) 
    print(f" {tp} {fp} {aa}  {matches} recall={recall} precision={precision} total matching time={end_time-start_time} seconds.")
    return recall, precision


if __name__ == '__main__':
  model="mpnet"
  print(f"./data/imdb_{model}.pqt")
  df11 = pd.read_parquet(f"./data/imdb_{model}.pqt")
  df22 = pd.read_parquet(f"./data/dbpedia_{model}.pqt")

  #import utilities
  #import pickle
  #df11['title_minhash'] = df11['title'].apply(utilities.get_minhash)
  #df22['title_minhash'] = df22['title'].apply(utilities.get_minhash)
  #df11['starring_minhash'] = df11['starring'].apply(utilities.get_minhash)
  #df22['starring_minhash'] = df22['actor name'].apply(utilities.get_minhash)
  #df11['title_minhash_bytes'] = df11['title_minhash'].apply(pickle.dumps)
  #df11['starring_minhash_bytes'] = df11['starring_minhash'].apply(pickle.dumps)
  #df22['title_minhash_bytes'] = df22['title_minhash'].apply(pickle.dumps)
  #df22['starring_minhash_bytes'] = df22['starring_minhash'].apply(pickle.dumps)
  #df11.drop(columns=["title_minhash","starring_minhash"]).to_parquet(f"./data/imdb_{model}.pqt", engine='pyarrow')  
  #df22.drop(columns=["title_minhash","starring_minhash"]).to_parquet(f"./data/dbpedia_{model}.pqt", engine='pyarrow')
  #exit(1)
  
  

  df11 = df11.dropna(subset=['title']) 
  df22 = df22.dropna(subset=['title'])


  df11['id'] = pd.to_numeric(df11['id'], errors='coerce')
  df22['id'] = pd.to_numeric(df22['id'], errors='coerce')
  df11 = df11.dropna(subset=['title'])
  truth = pd.read_csv("./data/truth_imdb_dbpedia.csv", sep="|", encoding="utf-8", keep_default_na=False)
  valid_d1_ids = set(df11['id'].values)
  valid_d2_ids = set(df22['id'].values)
  mask_to_keep = truth['D1'].isin(valid_d1_ids) & truth['D2'].isin(valid_d2_ids)
  truth = truth[mask_to_keep].copy()
  recall, precision = run(df11, df22, truth, model, 0.13)
  print(recall, precision)

