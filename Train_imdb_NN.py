import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# Ensure tensorflow is installed if you are running this locally
# For Google Colab or similar environments, Keras is usually pre-installed.
# If not, you might need: pip install tensorflow
from tensorflow import keras
from tensorflow.keras import layers
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
import Model
import utilities

model="mpnet_ft"
name=f"imdb_dbpedia_{model}"

num_candidates = 5
X_data = []
y_data = []
X_features = []



df1 = pd.read_parquet(f"./data/imdb_{model}.pqt")
#df1 = pd.read_parquet(f"./data/imdb_tuned.pqt")

df1['id'] = pd.to_numeric(df1['id'], errors='coerce')
df1 = df1.dropna(subset=['title'])

df2 = pd.read_parquet(f"./data/dbpedia_{model}.pqt")
df2 = df2.dropna(subset=['title'])
#df2 = pd.read_parquet(f"./data/dbpedia_tuned.pqt")
df2['id'] = pd.to_numeric(df2['id'], errors='coerce')

gold_standard = pd.read_csv(f"./data/truth_imdb_dbpedia.csv", sep="|", encoding="utf-8", keep_default_na=False)
valid_d1_ids = set(df1['id'].values)
valid_d2_ids = set(df2['id'].values)
mask_to_keep = gold_standard['D1'].isin(valid_d1_ids) & gold_standard['D2'].isin(valid_d2_ids)
gold_standard = gold_standard[mask_to_keep].copy()

df1['title_minhash'] = df1['title_minhash_bytes'].apply(pickle.loads)
df1['starring_minhash'] = df1['starring_minhash_bytes'].apply(pickle.loads)
df2['title_minhash'] = df2['title_minhash_bytes'].apply(pickle.loads)
df2['starring_minhash'] = df2['starring_minhash_bytes'].apply(pickle.loads)

minhash_titles1 = {row['id']: row['title_minhash'] for index, row in df1.iterrows()}
minhash_titles2 = {row['id']: row['title_minhash'] for index, row in df2.iterrows()}
minhash_actors1 = {row['id']: row['starring_minhash'] for index, row in df1.iterrows()}
minhash_actors2 = {row['id']: row['starring_minhash'] for index, row in df2.iterrows()}
titles1 = {row['id']: row['title'] for index, row in df1.iterrows()}
titles2 = {row['id']: row['title'] for index, row in df2.iterrows()}

vectors1 = df1['v'].tolist()
vectors2 = df2['v'].tolist()
datav = np.array(vectors1).astype(np.float32)
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
for index, row in sampled_gold_standard.iterrows():
    id1 = int(row["D1"])
    id2 = int(row["D2"])
    if id1 in embeddings1 and id2 in embeddings2:

      emb1 = embeddings1[id1]
      emb2 = embeddings2[id2]

      minhash_title1 = minhash_titles1[id1]
      minhash_title2 = minhash_titles2[id2]
      minhash_actor1 = minhash_actors1[id1]
      minhash_actor2 = minhash_actors2[id2]
      j_distance1 = minhash_title1.jaccard(minhash_title2)
      j_distance2 = minhash_actor1.jaccard(minhash_actor2)
      j_similarity1 = 1 - j_distance1
      j_similarity2 = 1 - j_distance2
      title1 = titles1[id1]
      title2 = titles2[id2]
      jw = jellyfish.jaro_winkler_similarity(str(title1), str(title2))

      # Ensure embeddings are numpy arrays and then concatenate
      combined_embedding = np.concatenate((np.array(emb1), np.array(emb2) ))
      features = np.array([j_distance1, j_distance2, j_similarity1, j_similarity2, jw])
      X_data.append(combined_embedding)
      X_features.append(features)
      y_data.append(1) # Match

      distances, indices1 = faiss_db.search(np.array([emb2]), num_candidates)
      for ind, ds in zip(indices1[0], distances[0]):
            id1_ =ids1_[ind]
            if id1_ != id1:
                emb1 = embeddings1[id1_]

                minhash_title1 = minhash_titles1[id1_]
                minhash_actor1 = minhash_actors1[id1_]
                j_distance1 = minhash_title1.jaccard(minhash_title2)
                j_distance2 = minhash_actor1.jaccard(minhash_actor2)
                j_similarity1 = 1 - j_distance1
                j_similarity2 = 1 - j_distance2
                title1 = titles1[id1]
                jw = jellyfish.jaro_winkler_similarity(str(title1), str(title2))

                combined_embedding = np.concatenate((np.array(emb1), np.array(emb2) ))
                features = np.array([ j_distance1, j_distance2, j_similarity1, j_similarity2, jw])
                X_data.append(combined_embedding)
                X_features.append(features)
                y_data.append(0)
print(f"Gold standard pairs creation completed with {len(X_data)} pairs. Ground truth matches: {sum(y_data)}, Ground truth non-matches: {len(y_data) - sum(y_data)}.")

num_positive_pairs = num_samples_to_take   #len(y_data)
num_negative_to_generate = num_positive_pairs # Or some multiple

all_ids_1 = list(embeddings1.keys())
all_ids_2 = list(embeddings2.keys())

 # Create a set of gold standard pairs for quick lookup
gold_pairs_set = set(zip(gold_standard["D2"], gold_standard["D1"]))

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

      minhash_title1 = minhash_titles1[random_id1]
      minhash_title2 = minhash_titles2[random_id2]
      minhash_actor1 = minhash_actors1[random_id1]
      minhash_actor2 = minhash_actors2[random_id2]
      j_distance1 = minhash_title1.jaccard(minhash_title2)
      j_distance2 = minhash_actor1.jaccard(minhash_actor2)
      j_similarity1 = 1 - j_distance1
      j_similarity2 = 1 - j_distance2
      title1 = titles1[random_id1]
      title2 = titles2[random_id2]
      jw = jellyfish.jaro_winkler_similarity(str(title1), str(title2))

      combined_embedding = np.concatenate((np.array(emb1), np.array(emb2) ))
      features = np.array([j_distance1, j_distance2, j_similarity1, j_similarity2, jw])
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

