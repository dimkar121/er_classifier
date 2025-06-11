import pandas as pd
import numpy as np
import faiss
import time
from scipy.spatial.distance import jaccard
import re
import jellyfish

def extract_model(text):
    """
    Extracts potential model numbers using a list of regex patterns robust
    enough for both Abt and Buy datasets.

    Args:
        text (str): The product name or description.

    Returns:
        set: A set of unique potential model number strings found in the text.
    """
    if not isinstance(text, str):
        return set()

    # A list of patterns, ordered to catch the most specific cases first.
    patterns = [
        # Pattern 1: Catches mixed alpha-numeric codes like 'FS105NA', 'WET54G', or 'F3H982-10'
        # It requires at least one letter and one number.
        r'\b(?=[A-Z0-9-]*[A-Z])(?=[A-Z0-9-]*[0-9])[A-Z0-9-]{4,}\b',

        # Pattern 2: Catches purely numeric codes of 5 digits or more, like '706018' or '64327'
        # This prevents matching small numbers like '100' from '10/100'.
        r'\b\d{5,}\b'
    ]

    found_models = set()
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            found_models.add(match.upper())

    return found_models


def are_models_matching(models_set_a, models_set_b):
    """
    Compares two sets of model numbers to see if they have any elements in common.

    Args:
        models_set_a (set): A set of model numbers for the first product.
        models_set_b (set): A set of model numbers for the second product.

    Returns:
        int: 1 if there is at least one common model number, 0 otherwise.
    """
    # First, ensure that both inputs are valid sets and not empty.
    # The 'if set1 and set2' handles cases where one might be None or an empty set.
    if models_set_a and models_set_b:
        # Check if the sets are NOT disjoint (i.e., they have a non-empty intersection)
        if not models_set_a.isdisjoint(models_set_b):
            return 1  # This means they share at least one model number.

    # If one of the sets is empty or they have no common elements, it's not a match.
    return 0


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

def find_brand_in_text(text, brand_list):
    """
    Searches for a brand from a given list within a text string.

    Args:
        text (str): The product name or description to search within.
        brand_list (list): A list of brands to look for.

    Returns:
        str: The found brand, or None if no brand is found.
    """
    text_lower = str(text).lower()
    for brand in brand_list:
        # Use regex to find the brand as a whole word to avoid partial matches
        # (e.g., matching 'on' in 'Sony')
        if re.search(r'\b' + re.escape(brand) + r'\b', text_lower):
            return brand
    return None


def check_brand_match(abt_brand, buy_brand):
    """Compares the extracted abt brand with the buy manufacturer column."""
    b1 = str(abt_brand).lower().strip()
    b2 = str(buy_brand).lower().strip()

    # Check that brands were found and that they match
    return 1 if b1 != 'none' and b2 != 'none' and b1 == b2 else 0



if __name__ == '__main__':
    truth = pd.read_csv("./data/truth_abt_buy.csv", sep=",", encoding="unicode_escape", keep_default_na=False)
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

    loaded_model_path = './data/er_abt_buy.keras'
    # Load the model
    loaded_model = keras.models.load_model(loaded_model_path)

    #loaded_model = xgb.XGBClassifier()
    #loaded_model.load_model("./data/abt_buy_xgb_model.json")


    print("Model loaded successfully!")
    # You can verify the model architecture
    #loaded_model.summary()
    batch_size = 10_000
    model = "mini"
    num_candidates = 5
    d = 384
    phi = 0.1520531820505105065205350
    df11 = pd.read_parquet(f"./data/Abt_embedded_mini_ft.pqt")
    df22 = pd.read_parquet(f"./data/Buy_embedded_mini_ft.pqt")
    vectors_buy = df22['v'].tolist()
    buy_embeddings = np.array(vectors_buy).astype(np.float32)
    vectors_abt = df11['v'].tolist()
    abt_embeddings = np.array(vectors_abt).astype(np.float32)
    abt_ids = np.array(df11['id'].tolist())
    buy_ids = np.array(df22['id'].tolist())

    df1_minhash = pd.read_parquet(f"./data/Abt_embedded_minhash_all.pqt")
    df2_minhash = pd.read_parquet(f"./data/Buy_embedded_minhash_all.pqt")
    minhash_names1 = {row['id']: row['namev'] for index, row in df1_minhash.iterrows()}
    minhash_names2 = {row['id']: row['namev'] for index, row in df2_minhash.iterrows()}
    minhash_descrs1 = {row['id']: row['descriptionv'] for index, row in df1_minhash.iterrows()}
    minhash_descrs2 = {row['id']: row['descriptionv'] for index, row in df2_minhash.iterrows()}
    df1_minhash['models'] = df1_minhash['name'].apply(extract_model)
    df2_minhash['models'] = df2_minhash['name'].apply(extract_model)
    models1 = {row['id']: row['models'] for index, row in df1_minhash.iterrows()}
    models2 = {row['id']: row['models'] for index, row in df2_minhash.iterrows()}
    names1 = {row['id']: row['name'] for index, row in df1_minhash.iterrows()}
    names2 = {row['id']: row['name'] for index, row in df2_minhash.iterrows()}
    prices1 = {row['id']: row['price'] for index, row in df1_minhash.iterrows()}
    prices2 = {row['id']: row['price'] for index, row in df2_minhash.iterrows()}
    all_brands = df2_minhash['brand'].dropna().unique()
    brands_list = sorted([str(b).lower() for b in all_brands if len(str(b)) > 2], key=len, reverse=True)
    df1_minhash['brand'] = df1_minhash['name'].apply(lambda text: find_brand_in_text(text, brands_list))
    brands1 = {row['id']: row['brand'] for index, row in df1_minhash.iterrows()}
    brands2 = {row['id']: row['brand'] for index, row in df2_minhash.iterrows()}

    index = faiss.IndexHNSWFlat(d, 32)
    index.hnsw.efConstruction = 60
    index.hnsw.efSearch = 16
    index.add(abt_embeddings)


     # thr.opt_inner_threshold(df22, df11, truth)


    tp = 0
    fp = 0
    start_time = time.time()


    for i in range(0, len(vectors_buy), batch_size):

        buys = buy_embeddings[i: i + batch_size]
        buy_ids_in_batch = buy_ids[i: i + batch_size]
        distances, candidate_indices = index.search(buys, num_candidates)  # return scholars

        '''
        repeated_buy_ids = np.repeat(buy_ids_in_batch, num_candidates)
        flat_candidate_ids = candidate_indices.flatten()
        abt_ids_in_batch = abt_ids[flat_candidate_ids]
        for abtId, buyId in zip(abt_ids_in_batch, repeated_buy_ids):
          tpFound = False
          if abtId in truthD.keys():
            idBuys = truthD[abtId]
            for idBuy in idBuys:
                if idBuy == buyId:
                    tp += 1
                    tpFound = True
                else:
                   fp += 1

        continue
        '''

        #print("candidates returned shape:", candidate_ids_batch.shape)

        # First, flatten the 2D array of IDs into a single long 1D array.
        # Shape changes from (1024, 5) to (5120,)
        flat_candidate_ids = candidate_indices.flatten()


        # Now, use this flat list of indices to grab all the corresponding DBLP
        # embeddings in a single, efficient operation.
        candidate_abt_embeddings = abt_embeddings[flat_candidate_ids]
        # We need to match each scholar embedding with its top-k candidates.
        # The np.repeat() function is perfect for this. It repeats each row of
        # scholar_batch k times.
        repeated_buy_embeddings = np.repeat(buys, num_candidates, axis=0)
        repeated_buy_ids = np.repeat(buy_ids_in_batch, num_candidates)


        features_list = []
        abt_ids_in_batch = abt_ids[flat_candidate_ids]
        for abtId_, buyId_ in zip(abt_ids_in_batch, repeated_buy_ids):
            minhash_name1 = minhash_names1[abtId_]
            minhash_name2 = minhash_names2[buyId_]
            minhash_descr1 = minhash_descrs1[abtId_]
            minhash_descr2 = minhash_descrs2[buyId_]
            j_distance1 = jaccard(minhash_name1, minhash_name2)
            j_distance2 = jaccard(minhash_descr1, minhash_descr2)
            j_similarity1 = 1 - j_distance1
            j_similarity2 = 1 - j_distance2

            model1 = models1[abtId_]
            model2 = models2[buyId_]
            name1 = names1[abtId_]
            name2 = names2[buyId_]
            price1 = prices1[abtId_]
            price2 = prices2[buyId_]
            brand1 = brands1[abtId_]
            brand2 = brands2[buyId_]
            br = check_brand_match(brand1, brand2)
            jw = jellyfish.jaro_winkler_similarity(str(name1), str(name2))
            price_diff = calculate_price_diff(price1, price2)
            m = are_models_matching(model1, model2)

            features_list.append([j_similarity1, j_similarity2, m, price_diff, jw])
        features_array = np.array(features_list, dtype='float32')
        #features_2d = features_array.reshape(-1, 1)


        # Now, combine the two embedding arrays side-by-side.
        # This creates the final "mega-batch" for your neural network.
        combined_embeddings = np.concatenate([candidate_abt_embeddings, repeated_buy_embeddings], axis=1)

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

        #combined_embeddings = features_array
        #print(combined_embeddings.shape)

        predictions = loaded_model.predict( [combined_embeddings, interactions, features_array], verbose=0)

        #predictions = loaded_model.predict_proba(combined_embeddings)


        #print(f"Predictions shape {predictions.flatten().shape}  candidates  shape {candidate_indices.flatten().shape} {repeated_scholar_ids.shape} ")
        predicted_statuses = (predictions > phi).astype(int).flatten()

        for predicted_status, abt_ind, buyId in zip(predicted_statuses, candidate_indices.flatten(), repeated_buy_ids):

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


    end_time = time.time()
    print(f"recall={round(tp / matches, 2)} precision={round(tp / (tp + fp), 2)} total matching time={end_time-start_time} seconds.")
