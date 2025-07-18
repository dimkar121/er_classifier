import numpy as np
import pandas as pd
import math
import faiss
import Model

model = "mpnet_ft"
name = f"fodors_zagats_{model}"
id1df = "idFodors"
id2df = "idZagats"
num_candidates = 5
X_data = []
y_data = []
X_features = []



df1 = pd.read_parquet(f"./data/fodors_{model}.pqt")
df2 = pd.read_parquet(f"./data/zagats_{model}.pqt")
gold_standard = pd.read_csv(f"./data/truth_fodors_zagats.csv", sep=",", encoding="utf-8", keep_default_na=False)
vectors1 = df1['v'].tolist()
vectors2 = df2['v'].tolist()
d = 384
datav = np.array(vectors1).astype(np.float32)
d = datav.shape[1]
faiss_db = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
faiss_db.hnsw.efConstruction = 60
faiss_db.hnsw.efSearch = 64

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
  id1 = row[id1df]
  id2 = row[id2df]

  if id1 in embeddings1 and id2 in embeddings2:
        emb1 = embeddings1[id1]
        emb2 = embeddings2[id2]

        combined_embedding = np.concatenate((np.array(emb1), np.array(emb2) ))
        X_data.append(combined_embedding)
        y_data.append(1) # Match

        distances, indices1 = faiss_db.search(np.array([emb2]), num_candidates)
        for ind, ds in zip(indices1[0], distances[0]):
              id1_ =ids1_[ind]
              if id1_ != id1:
                  emb1 = embeddings1[id1_]
                  combined_embedding = np.concatenate((np.array(emb1), np.array(emb2) ))
                  X_data.append(combined_embedding)
                  y_data.append(0)
print(f"Gold standard pairs creation completed with {len(X_data)} pairs. Ground truth matches: {sum(y_data)}, Ground truth non-matches: {len(y_data) - sum(y_data)}.")

num_positive_pairs = num_samples_to_take   #len(y_data)
num_negative_to_generate = num_positive_pairs # Or some multiple

all_ids_1 = list(embeddings1.keys())
all_ids_2 = list(embeddings2.keys())

# Create a set of gold standard pairs for quick lookup
gold_pairs_set = set(zip(gold_standard[id2df], gold_standard[id1df]))

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

        combined_embedding = np.concatenate((np.array(emb1), np.array(emb2) ))
        X_data.append(combined_embedding)
        y_data.append(0) # Non-match
        generated_negatives += 1




print(f"Training data was generated. Ground truth matches: {sum(y_data)}, Ground truth non-matches: {len(y_data) - sum(y_data)}.")


X_data = np.array(X_data)
#X_features = np.array(X_features)
y = np.array(y_data)

emb1_batch = X_data[:, :d]
emb2_batch = X_data[:, d:]

numerator = np.einsum('ij,ij->i', emb1_batch, emb2_batch)
denominator = np.linalg.norm(emb1_batch, axis=1) * np.linalg.norm(emb2_batch, axis=1)
epsilon = 1e-7
cosine_sim_scores = (numerator / (denominator + epsilon)).reshape(-1, 1)
X_features =  cosine_sim_scores


diff_vectors = emb1_batch - emb2_batch
product_vectors = emb1_batch * emb2_batch
X_interactions = np.concatenate([diff_vectors, product_vectors], axis=1)

Model.train(name, X_data, X_features, X_interactions, y)

