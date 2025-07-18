import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# Ensure tensorflow is installed if you are running this locally
# For Google Colab or similar environments, Keras is usually pre-installed.
# If not, you might need: pip install tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, Subtract, Multiply

from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import math
import faiss
from scipy.spatial.distance import jaccard
import re
import jellyfish
import utilities
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from itertools import product
import Model
import pickle

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


# --- 2. Category Similarity Feature ---

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


model="mpnet"
name=f"amazon_walmart_{model}"

num_candidates = 5
X_data = []
y_data = []
X_features = []


df2 = pd.read_parquet(f"./data/walmart_products_{model}.pqt")
df2['id'] = pd.to_numeric(df2['id'], errors='coerce')
df2.dropna(subset=['id'], inplace=True)
df2['id'] = df2['id'].astype(int)

df1 = pd.read_parquet(f"./data/amazon_products_{model}.pqt")
df1['id'] = pd.to_numeric(df1['id'], errors='coerce')
df1.dropna(subset=['id'], inplace=True)
df1['id'] = df1['id'].astype(int)

gold_standard = pd.read_csv(f"./data/truth_amazon_walmart.tsv", sep="\t", encoding="utf-8", keep_default_na=False)
gold_standard['id1'] = gold_standard['id1'].astype(int)
gold_standard['id2'] = gold_standard['id2'].astype(int)

df2.reset_index(drop=True, inplace=True)
df1.reset_index(drop=True, inplace=True)


df1['title_minhash'] = df1['title_bytes'].apply(pickle.loads)
df2['title_minhash'] = df2['title_bytes'].apply(pickle.loads)
minhash_titles1 = {row['id']: row['title_minhash'] for index, row in df1.iterrows()}
minhash_titles2 = {row['id']: row['title_minhash'] for index, row in df2.iterrows()}

titles1 = {row['id']: row['title'] for index, row in df1.iterrows()}
titles2 = {row['id']: row['title'] for index, row in df2.iterrows()}
vectors1 = df1['v'].tolist()
vectors2 = df2['v'].tolist()
datav = np.array(vectors2).astype(np.float32)
d = datav.shape[1]
faiss_db = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
faiss_db.hnsw.efConstruction = 60
faiss_db.hnsw.efSearch = 16
faiss_db.add(datav)
ids1_ = df1['id'].tolist()
ids2_ = df2['id'].tolist()

# Create dictionaries for quick embedding lookups
embeddings1 = {row['id']: row['v'] for index, row in df1.iterrows()}
embeddings2 = {row['id']: row['v'] for index, row in df2.iterrows()}

num_samples_to_take = math.ceil(len(gold_standard) * 0.60)
print(f"{num_samples_to_take} pairs will be sampled.")
sampled_gold_standard = gold_standard.sample(n=num_samples_to_take, random_state=42, replace=False)
# Positive Pairs
for index, row in sampled_gold_standard.iterrows():
    id_amazon = int(row["id1"])
    id_walmart = int(row["id2"])
    if not id_amazon in df1['id'].values or not id_walmart in df2['id'].values:
        print("Either of these is orphan", id_amazon, id_walmart)
        continue

    if id_amazon in embeddings1 and id_walmart in embeddings2:
      emb1 = embeddings1[id_amazon]
      emb2 = embeddings2[id_walmart]

      minhash_title1 = minhash_titles1[id_amazon]
      minhash_title2 = minhash_titles2[id_walmart]
      j_distance1 = minhash_title1.jaccard(minhash_title2)
      j_similarity1 = 1 - j_distance1
      title1 = titles1[id_amazon]
      title2 = titles2[id_walmart]
      jw = jellyfish.jaro_winkler_similarity(str(title1), str(title2))
      combined_embedding = np.concatenate((np.array(emb1), np.array(emb2) ))
      #features = np.array([ j_similarity1,  model_sim, model_match, brand_sim, brand_m])
      features = np.array([j_similarity1, jw])

      # amazon-google features = np.array([j_similarity1, j_similarity2, m, price_diff, jw])
      X_data.append(combined_embedding)
      X_features.append(features)
      y_data.append(1) # Match

      distances, indices1 = faiss_db.search(np.array([emb1]), num_candidates)
      for ind, ds in zip(indices1[0], distances[0]):
            id_walmart_ =ids2_[ind]
            if id_walmart_ != id_walmart and id_walmart_ in embeddings2:
                emb2 = embeddings2[id_walmart_]
                minhash_title2 = minhash_titles2[id_walmart_]
                j_distance1 = minhash_title1.jaccard(minhash_title2)
                j_similarity1 = 1 - j_distance1
                title2 = titles2[id_walmart_]
                jw = jellyfish.jaro_winkler_similarity(str(title1), str(title2))
                combined_embedding = np.concatenate((np.array(emb1), np.array(emb2) ))
                #features = np.array([ j_similarity1,  model_sim, model_match, brand_sim, brand_m])
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
gold_pairs_set = set(zip(gold_standard["id2"], gold_standard["id1"]))

generated_negatives = 0
max_attempts = num_negative_to_generate * 5 # To avoid infinite loop

print(f"About to generate {num_negative_to_generate} non-matches with max. {max_attempts} attempts.")
attempts = 0
while generated_negatives < num_negative_to_generate and attempts < max_attempts:
  attempts += 1
  random_id1 = np.random.choice(all_ids_1)
  random_id2 = np.random.choice(all_ids_2)

  if (random_id2, random_id1) not in gold_pairs_set and random_id1 in embeddings1 and  random_id2 in embeddings2:
      emb1 = embeddings1[random_id1]
      emb2 = embeddings2[random_id2]

      minhash_title1 = minhash_titles1[random_id1]
      minhash_title2 = minhash_titles2[random_id2]
      j_distance1 = minhash_title1.jaccard( minhash_title2)
      j_similarity1 = 1 - j_distance1
      title1 = titles1[random_id1]
      title2 = titles2[random_id2]
      jw = jellyfish.jaro_winkler_similarity(str(title1), str(title2))

      combined_embedding = np.concatenate((np.array(emb1), np.array(emb2) ))
      #features = np.array([j_similarity1, model_sim, model_match, brand_sim, brand_m])
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


