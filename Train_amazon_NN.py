import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# Ensure tensorflow is installed if you are running this locally
# For Google Colab or similar environments, Keras is usually pre-installed.
# If not, you might need: pip install tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import math
import faiss
from scipy.spatial.distance import jaccard
import re
#import mmh3
import pickle
import jellyfish
from sklearn.metrics import precision_score, recall_score, f1_score
import Model


model="mpnet"
name=f"amazon_google_{model}"


num_candidates = 5
X_data = []
y_data = []
X_features = []





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


df1 = pd.read_parquet(f"./data/Amazon_{model}.pqt")
df2 = pd.read_parquet(f"./data/Google_{model}.pqt")
gold_standard = pd.read_csv(f"./data/truth_amazon_google.csv", sep=",", encoding="utf-8", keep_default_na=False)


df1['title_minhash'] = df1['title_bytes'].apply(pickle.loads)
df2['name_minhash'] = df2['name_bytes'].apply(pickle.loads)
minhash_names1 = {row['id']: row['title_minhash'] for index, row in df1.iterrows()}
minhash_names2 = {row['id']: row['name_minhash'] for index, row in df2.iterrows()}
#df1['models'] = df1['title'].apply(extract_model)
#df2['models'] = df2['name'].apply(extract_model)
#models1 = {row['id']: row['models'] for index, row in df1.iterrows()}
#models2 = {row['id']: row['models'] for index, row in df2.iterrows()}
names1 = {row['id']: row['title_minhash'] for index, row in df1.iterrows()}
names2 = {row['id']: row['name_minhash'] for index, row in df2.iterrows()}
#prices1 = {row['id']: row['price'] for index, row in df1.iterrows()}
#prices2 = {row['id']: row['price'] for index, row in df2.iterrows()}
# all_brands = df1['manufacturer'].dropna().unique()
# brands_list = sorted([str(b).lower() for b in all_brands if len(str(b)) > 2], key=len, reverse=True)
#df1_minhash['brand'] = df1_minhash['name'].apply(lambda text: find_brand_in_text(text, brands_list))
# brands1 = {row['id']: row['manufacturer'] for index, row in df1.iterrows()}
# df2['brand'] = df2['name'].apply(lambda text: find_brand_in_text(text, brands_list))
# brands2 = {row['id']: row['brand'] for index, row in df2.iterrows()}

vectors1 = df1['v'].tolist()
vectors2 = df2['v'].tolist()

datav = np.array(vectors1).astype(np.float32)
d = datav.shape[1]
faiss_db = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
faiss_db.hnsw.efConstruction = 60
faiss_db.hnsw.efSearch = 16
datav = np.array(vectors1).astype(np.float32)
faiss_db.add(datav)
ids1_ = df1['id'].tolist()

# Create dictionaries for quick embedding lookups
embeddings1 = {row['id']: row['v'] for index, row in df1.iterrows()}
embeddings2 = {row['id']: row['v'] for index, row in df2.iterrows()}


num_samples_to_take = math.ceil(len(gold_standard) * 0.60)
print(f"{num_samples_to_take} pairs will be sampled.")
sampled_gold_standard = gold_standard.sample(n=num_samples_to_take, random_state=42, replace=False)
# Positive Pairs
for index, row in sampled_gold_standard.iterrows():
    id1 = row["idAmazon"]
    id2 = row["idGoogleBase"]

    if id1 in embeddings1 and id2 in embeddings2:
      emb1 = embeddings1[id1]
      emb2 = embeddings2[id2]


      minhash_name1 = minhash_names1[id1]
      minhash_name2 = minhash_names2[id2]

      j_distance1 = minhash_name1.jaccard(minhash_name2)
      j_similarity1 = 1 - j_distance1
      #model1 = models1[id1]
      #model2 = models2[id2]
      name1 = names1[id1]
      name2 = names2[id2]
      #price1 = prices1[id1]
      #price2 = prices2[id2]
      #brand1 = brands1[id1]
      #brand2 = brands2[id2]
      #br = check_brand_match(brand1, brand2)
      jw = jellyfish.jaro_winkler_similarity(str(name1), str(name2))
      #price_diff = calculate_price_diff(price1, price2)
      #m = are_models_matching(model1, model2)
      # Ensure embeddings are numpy arrays and then concatenate

      combined_embedding = np.concatenate((np.array(emb1), np.array(emb2) ))
      #features = np.array([j_similarity1, j_similarity2, m, price_diff, jw])
      features = np.array([j_similarity1, jw])

      X_data.append(combined_embedding)
      X_features.append(features)
      y_data.append(1) # Match
      distances, indices1 = faiss_db.search(np.array([emb2]), num_candidates)
      for ind, ds in zip(indices1[0], distances[0]):
              id1_ =ids1_[ind]
              if id1_ != id1:
                  emb1 = embeddings1[id1_]

                  minhash_name1 = minhash_names1[id1_]
                  #minhash_descr1 = minhash_descrs1[id1_]
                  #model1 = models1[id1_]
                  name1 = names1[id1_]
                  #price1 = prices1[id1_]
                  #brand1 = brands1[id1_]
                  #br = check_brand_match(brand1, brand2)
                  j_distance1 = minhash_name1.jaccard(minhash_name2)
                  j_similarity1 = 1 - j_distance1
                  #m = are_models_matching(model1, model2)
                  #price_diff = calculate_price_diff(price1, price2)
                  jw = jellyfish.jaro_winkler_similarity(str(name1), str(name2))

                  combined_embedding = np.concatenate((np.array(emb1), np.array(emb2) ))
                  features = np.array([j_similarity1,  jw])
                  X_data.append(combined_embedding)
                  X_features.append(features)
                  y_data.append(0)
print(f"Gold standard pairs creation completed with {len(X_data)} pairs. Ground truth matches: {sum(y_data)}, Ground truth non-matches: {len(y_data) - sum(y_data)}.")

num_positive_pairs = num_samples_to_take   #len(y_data)
num_negative_to_generate = num_positive_pairs # Or some multiple

all_ids_1 = list(embeddings1.keys())
all_ids_2 = list(embeddings2.keys())

# Create a set of gold standard pairs for quick lookup
gold_pairs_set = set(zip(gold_standard["idGoogleBase"], gold_standard["idAmazon"]))

generated_negatives = 0
max_attempts = num_negative_to_generate * 5 # To avoid infinite loop

print(f"About to generate {num_negative_to_generate} non-matches with max. {max_attempts} attempts.")
attempts = 0
while generated_negatives < num_negative_to_generate and attempts < max_attempts:
  attempts += 1
  random_id1 = np.random.choice(all_ids_1)
  random_id2 = np.random.choice(all_ids_2)

  if (random_id2, random_id1) not in gold_pairs_set:
      emb1 = embeddings1[random_id1]
      emb2 = embeddings2[random_id2]

      minhash_name1 = minhash_names1[random_id1]
      minhash_name2 = minhash_names2[random_id2]
      #minhash_descr1 = minhash_descrs1[random_id1]
      #minhash_descr2 = minhash_descrs2[random_id2]
      #model1 = models1[random_id1]
      #model2 = models2[random_id2]
      #m = are_models_matching(model1, model2)
      name1 = names1[random_id1]
      name2 = names2[random_id2]
      #price1 = prices1[random_id1]
      #price2 = prices2[random_id2]
      #brand1 = brands1[random_id1]
      #brand2 = brands2[random_id2]
      #br = check_brand_match(brand1, brand2)
      jw = jellyfish.jaro_winkler_similarity(str(name1), str(name2))
      #price_diff = calculate_price_diff(price1, price2)
      j_distance1 = minhash_name1.jaccard(minhash_name2)
      j_similarity1 = 1 - j_distance1

      combined_embedding = np.concatenate((np.array(emb1), np.array(emb2) ))
      features = np.array([j_similarity1, jw])
      X_data.append(combined_embedding)
      X_features.append(features)
      y_data.append(0) # Non-match
      generated_negatives += 1


print(f"Training data was generated. Ground truth matches: {sum(y_data)}, Ground truth non-matches: {len(y_data) - sum(y_data)}.")

X_data = np.array(X_data)
X_features = np.array(X_features)
y = np.array(y_data)

emb1_batch = X_data[:, :d]
emb2_batch = X_data[:, d:]
numerator = np.einsum('ij,ij->i', emb1_batch, emb2_batch)
denominator = np.linalg.norm(emb1_batch, axis=1) * np.linalg.norm(emb2_batch, axis=1)
epsilon = 1e-7
cosine_sim_scores = (numerator / (denominator + epsilon)).reshape(-1, 1)
X_features = np.concatenate([X_features, cosine_sim_scores], axis=1)

diff_vectors = emb1_batch - emb2_batch
product_vectors = emb1_batch * emb2_batch
X_interactions = np.concatenate([diff_vectors, product_vectors], axis=1)

Model.train(name, X_data, X_features, X_interactions, y)




