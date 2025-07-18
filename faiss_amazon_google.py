import pandas as pd
import numpy as np
import faiss
import time
import re
import jellyfish
from scipy.spatial.distance import jaccard
import pickle


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


#if __name__ == '__main__':
def run(df11, df22, truth, model_name, phi):

    

    truthD = dict()
    a = 0
    for i, r in truth.iterrows():
        idAmazon = r["idAmazon"]
        idGoogle = r["idGoogleBase"]
        if idAmazon in truthD.keys():
            ids = truthD[idAmazon]
            ids.append(idGoogle)
            a += 1
        else:
            truthD[idAmazon] = [idGoogle]
            a += 1
    matches = a
    print("No of matches=", matches)

    # ====================================================================--

    from tensorflow import keras  # Or `import keras` depending on your setup

    #loaded_model_path = './data/er_abt_buy.keras'
    loaded_model_path = f'./data/er_amazon_google_{model_name}.keras'
    # use the er_abt model for both fine-tuned and non-fine-tuned embeddings
    # use the er_amazon for both  fine-tuned and non-fine-tuned embeddings
    # Load the model
    loaded_model = keras.models.load_model(loaded_model_path)
    print("Model loaded successfully!")
    # You can verify the model architecture
    loaded_model.summary()
    batch_size = 10_000
    num_candidates = 5

    vectors_google = df22['v'].tolist()
    google_embeddings = np.array(vectors_google).astype(np.float32)
    d = google_embeddings.shape[1]
    vectors_amazon = df11['v'].tolist()
    amazon_embeddings = np.array(vectors_amazon).astype(np.float32)
    amazon_ids = np.array(df11['id'].tolist())
    google_ids = df22['id'].tolist()
    df11['title_minhash'] = df11['title_bytes'].apply(pickle.loads)
    df22['title_minhash'] = df22['name_bytes'].apply(pickle.loads)

    minhash_names1 = {row['id']: row['title_minhash'] for index, row in df11.iterrows()}
    minhash_names2 = {row['id']: row['title_minhash'] for index, row in df22.iterrows()}
    #df1['models'] = df1['title'].apply(extract_model)
    #df1['models'] = df1['title'].apply(extract_model)
    #df2['models'] = df2['name'].apply(extract_model)
    #models1 = {row['id']: row['models'] for index, row in df1.iterrows()}
    #models2 = {row['id']: row['models'] for index, row in df2.iterrows()}
    #names1 = {row['id']: row['title'] for index, row in df1.iterrows()}
    names1 = {row['id']: row['title'] for index, row in df11.iterrows()}
    names2 = {row['id']: row['name'] for index, row in df22.iterrows()}
    #prices1 = {row['id']: row['price'] for index, row in df1.iterrows()}
    #prices2 = {row['id']: row['price'] for index, row in df2.iterrows()}
    #all_brands = df1['manufacturer'].dropna().unique()
    #brands_list = sorted([str(b).lower() for b in all_brands if len(str(b)) > 2], key=len, reverse=True)
    # df1_minhash['brand'] = df1_minhash['name'].apply(lambda text: find_brand_in_text(text, brands_list))
    #brands1 = {row['id']: row['manufacturer'] for index, row in df1.iterrows()}
    #brands2 = {row['id']: row['manufacturer'] for index, row in df2.iterrows()}
    #df2['brand'] = df2['name'].apply(lambda text: find_brand_in_text(text, brands_list))
    #brands2 = {row['id']: row['brand'] for index, row in df2.iterrows()}

    index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 60
    index.hnsw.efSearch = 16
    index.add(amazon_embeddings)

    tp = 0
    fp = 0
    start_time = time.time()

    for i in range(0, len(vectors_google), batch_size):

        googles = google_embeddings[i: i + batch_size]
        google_ids_in_batch = google_ids[i: i + batch_size]
        distances, candidate_indices = index.search(googles, num_candidates)  # return scholars
        #print("candidates returned shape:", candidate_ids_batch.shape)

        flat_candidate_ids = candidate_indices.flatten()
        candidate_amazon_embeddings = amazon_embeddings[flat_candidate_ids]
        repeated_google_embeddings = np.repeat(googles, num_candidates, axis=0)
        repeated_google_ids = np.repeat(google_ids_in_batch, num_candidates)

        features_list = []
        amazon_ids_in_batch = amazon_ids[flat_candidate_ids]
        for amazonId_, googleId_ in zip(amazon_ids_in_batch, repeated_google_ids):
            minhash_name1 = minhash_names1[amazonId_]
            minhash_name2 = minhash_names2[googleId_]
            #minhash_descr1 = minhash_descrs1[amazonId_]
            #minhash_descr2 = minhash_descrs2[googleId_]
            j_distance1 = minhash_name1.jaccard( minhash_name2)
            #j_distance2 = jaccard(minhash_descr1, minhash_descr2)
            j_similarity1 = 1 - j_distance1
            #j_similarity2 = 1 - j_distance2

            #model1 = models1[amazonId_]
            #model2 = models2[googleId_]
            #print("models", model1, model2)
            name1 = names1[amazonId_]
            name2 = names2[googleId_]
            #price1 = prices1[amazonId_]
            #price2 = prices2[googleId_]
            #brand1 = brands1[amazonId_]
            #brand2 = brands2[googleId_]
            #print("brands", brand1," ------ ", brand2)
            #br = check_brand_match(brand1, brand2)
            jw = jellyfish.jaro_winkler_similarity(str(name1), str(name2))
            #price_diff = calculate_price_diff(price1, price2)
            #m = are_models_matching(model1, model2)

            features_list.append([j_similarity1, jw])
        features_array = np.array(features_list, dtype='float32')
        combined_embeddings = np.concatenate([candidate_amazon_embeddings, repeated_google_embeddings], axis=1)
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
        predictions = loaded_model.predict([combined_embeddings,interactions, features_array], verbose=0)
        predicted_statuses = (predictions > phi).astype(int).flatten()

        for predicted_status, amazon_ind, googleId in zip(predicted_statuses, candidate_indices.flatten(), repeated_google_ids):

          if predicted_status == 1:
            amazonId = amazon_ids[amazon_ind]
            if amazonId in truthD.keys():
                tpFound = False
                idGoogles = truthD[amazonId]
                for idGoogle in idGoogles:
                    if idGoogle == googleId:
                        tp += 1
                        tpFound = True
                if tpFound == False:
                    fp += 1


    end_time = time.time()
    recall = round(tp / matches, 2)
    precision = round(tp / (tp + fp), 2)
    return recall, precision


if __name__ == '__main__':
  model="mini_ft"
  df11 = pd.read_parquet(f"./data/Amazon_{model}.pqt")
  df22 = pd.read_parquet(f"./data/Google_{model}.pqt")
  # Read only the schema from the Parquet file
  #import pyarrow.parquet as pq 
  #schema = pq.read_schema(f"./data/Amazon_{model}.pqt")
  # Get the list of column names
  #column_names = schema.names
  #print(column_names)
  

  truth = pd.read_csv("./data/truth_amazon_google.csv", sep=",", encoding="utf-8", keep_default_na=False)
  recall, precision = run(df11, df22, truth, model, 0.15)
  print(recall, precision)



