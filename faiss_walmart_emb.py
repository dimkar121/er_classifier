import pandas as pd
import numpy as np
import faiss
import time
from scipy.spatial.distance import jaccard
import re
import jellyfish
from itertools import product

def calculate_modelno_similarity(modelno1, modelno2):
  """
  Calculates the Jaro-Winkler similarity score between two model number strings.
  """
  # Convert to string and handle potential missing values
  str1 = str(modelno1) if pd.notna(modelno1) else ""
  str2 = str(modelno2) if pd.notna(modelno2) else ""
  if not str1 or not str2:
     return 0.0
  return jellyfish.jaro_winkler_similarity(str1, str2)



def calculate_category_jaccard(cat1, cat2):
    """
    Calculates the Jaccard similarity of the words in the category strings.
    """
    # Simple cleaning: lowercase and split
    set1 = set(str(cat1).lower().split())
    set2 = set(str(cat2).lower().split())

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    return intersection / union if union > 0 else 0.0


# --- 3. Differing Tokens Count Feature ---

def count_differing_tokens(title1, title2):
    """
    Counts the number of unique words that are in one title but not the other.
    A higher count can indicate a non-match.
    """
    set1 = set(str(title1).lower().split())
    set2 = set(str(title2).lower().split())

    # Calculate the symmetric difference
    return len(set1.symmetric_difference(set2))

def model_number_match(modelno1, modelno2):
    """
    Checks if two model number strings are identical after cleaning.
    Returns 1 for a match, 0 otherwise.
    """
    # Convert to string, clean, and handle potential missing values
    str1 = str(modelno1).lower().strip() if pd.notna(modelno1) else ""
    str2 = str(modelno2).lower().strip() if pd.notna(modelno2) else ""

    # Return 1 only if both strings are valid and they are identical
    if str1 and str2 and str1 == str2:
        return 1
    return 0


def brand_match(brand1, brand2):
    """
    Checks if two brand strings are identical after cleaning.
    Returns 1 for a match, 0 otherwise.
    """
    str1 = str(brand1).lower().strip() if pd.notna(brand1) else ""
    str2 = str(brand2).lower().strip() if pd.notna(brand2) else ""

    if str1 and str2 and str1 == str2:
        return 1
    return 0


def calculate_brand_similarity(brand1, brand2):
    """
    Calculates the Jaro-Winkler similarity score between two brand strings.
    """
    str1 = str(brand1) if pd.notna(brand1) else ""
    str2 = str(brand2) if pd.notna(brand2) else ""

    if not str1 or not str2:
        return 0.0

    return jellyfish.jaro_winkler_similarity(str1, str2)

def calculate_price_diff(price1, price2):
    """Calculates the normalized difference between two prices."""
    # Handle cases where price might be missing (None, NaN) or zero
    try:
        p1 = float(price1)
        p2 = float(price2)
    except (ValueError, TypeError):
        return 0  # Return 0 if prices are not valid numbers

    if max(p1, p2) == 0:
        return 0

    return abs(p1 - p2) / max(p1, p2)

if __name__ == '__main__':
    df22 = pd.read_parquet(f"./data/walmart_products_tuned.pqt")
    df11 = pd.read_parquet(f"./data/amazon_products_tuned.pqt")
    #df22 = pd.read_parquet(f"./data/walmart_products.pqt")
    #df11 = pd.read_parquet(f"./data/amazon_products.pqt")

    df11['id'] = pd.to_numeric(df11['id'], errors='coerce')
    df11.dropna(subset=['id'], inplace=True)
    df11['id'] = df11['id'].astype(int)

    df22['id'] = pd.to_numeric(df22['id'], errors='coerce')
    df22.dropna(subset=['id'], inplace=True)
    df22['id'] = df22['id'].astype(int)

    truth = pd.read_csv("./data/truth_amazon_walmart.tsv", sep="\t", encoding="unicode_escape", keep_default_na=False)
    truth['id1'] = truth['id1'].astype(int)
    truth['id2'] = truth['id2'].astype(int)
    truthD = dict()
    a = 0
    for i, r in truth.iterrows():
        id_walmart = r["id2"] #id2
        id_amazon = r["id1"]  #id1
        if not id_walmart in df22['id'].values or not id_amazon in df11['id'].values:
                print(f"Disregarding {id_walmart} or {id_amazon}")
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
    loaded_model_path = './data/er_abt_buy.keras'
    # Load the model
    loaded_model = keras.models.load_model(loaded_model_path)


    print("Model loaded successfully!")
    batch_size = 10_000
    num_candidates = 5
    d = 384
    phi = 0.80

    minhash_titles1 = {row['id']: row['title_v'] for index, row in df11.iterrows()}
    minhash_titles2 = {row['id']: row['title_v'] for index, row in df22.iterrows()}
    minhash_models1 = {row['id']: row['modelno_v'] for index, row in df11.iterrows()}
    minhash_models2 = {row['id']: row['modelno_v'] for index, row in df22.iterrows()}
    titles1 = {row['id']: row['title'] for index, row in df11.iterrows()}
    titles2 = {row['id']: row['title'] for index, row in df22.iterrows()}
    categories1 = {row['id']: row['category'] for index, row in df11.iterrows()}
    categories2 = {row['id']: row['category'] for index, row in df22.iterrows()}
    brands1 = {row['id']: row['brand'] for index, row in df11.iterrows()}
    brands2 = {row['id']: row['brand'] for index, row in df22.iterrows()}
    models1 = {row['id']: row['modelno'] for index, row in df11.iterrows()}
    models2 = {row['id']: row['modelno'] for index, row in df22.iterrows()}
    prices1 = {row['id']: row['price'] for index, row in df11.iterrows()}
    prices2 = {row['id']: row['price'] for index, row in df22.iterrows()}

    vectors_amazon = df11['v'].tolist()   #df11
    amazon_embeddings = np.array(vectors_amazon).astype(np.float32)
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
            minhash_model1 = minhash_models1[amazonId_]
            minhash_model2 = minhash_models2[walmartId_]
            j_distance1 = jaccard(minhash_title1, minhash_title2)
            j_distance2 = jaccard(minhash_model1, minhash_model2)
            j_similarity1 = 1 - j_distance1
            j_similarity2 = 1 - j_distance2
            title1 = titles1[amazonId_]
            title2 = titles2[walmartId_]
            price1 = prices1[amazonId_]
            price2 = prices2[walmartId_]
            jw = jellyfish.jaro_winkler_similarity(str(title1), str(title2))
            category1 = categories1[amazonId_]
            category2 = categories2[walmartId_]
            jw2 = jellyfish.jaro_winkler_similarity(str(category1), str(category2))
            brand1 = brands1[amazonId_]
            brand2 = brands2[walmartId_]
            jw3 = jellyfish.jaro_winkler_similarity(str(brand1), str(brand2))
            model1 = models1[amazonId_]
            model2 = models2[walmartId_]
            model_match = model_number_match(model1, model2)
            model_sim = calculate_modelno_similarity(model1, model2)
            cat_jac = calculate_category_jaccard(category1, category2)
            tokens = count_differing_tokens(title1, title2)
            brand_sim= calculate_brand_similarity(brand1, brand2)
            brand_m = brand_match(brand1, brand2)
            price_diff = calculate_price_diff(price1, price2)
            #features_list.append([ j_similarity1, model_sim, model_match, brand_sim, brand_m ])
            features_list.append([ j_similarity1,  model_match, price_diff, jw])

        features_array = np.array(features_list, dtype='float32')


        combined_embeddings = np.concatenate([candidate_walmart_embeddings, repeated_amazon_embeddings], axis=1)

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
    print(f" {tp} recall={round(tp / matches, 2)} precision={round(tp / (tp + fp), 2)} total matching time={end_time-start_time} seconds.")
