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

name="walmart_amazon"
files1 =["walmart_products"] #["votersA"] # ["Scholar"] #["Abt"] #["Scholar"] #["Amazon", "Scholar", "votersA"] #["Scholar"] # ["Amazon", "Scholar", "votersA"]   #"Abt" # "Scholar"  #"Abt" # "ACM" #
files2 = ["amazon_products"] #["votersB"] #["DBLP2"] # ["Buy"] #["DBLP2"] #["Google", "DBLP2" , "votersB"] #["DBLP2"] #["Google", "DBLP2" , "votersB"]  # "Buy" #"DBLP2" # #"DBLP" #
files3 =["amazon_walmart"] # ["voters"] #["Scholar_DBLP"] # ["abt_buy"] #["Scholar_DBLP"] #["Amazon_googleProducts", "Scholar_DBLP", "voters"] #["Scholar_DBLP"] #["Amazon_googleProducts", "Scholar_DBLP", "voters"]# "abt_buy" #"Scholar_DBLP" # # "ACM_DBLP" #
id1dfs = ["id1"] #["id1"] # ["idScholar"] # ["idAbt"] #["idScholar"] # ["idAmazon", "idScholar", "id1"] #["idScholar"] #["idAmazon", "idScholar", "id1"] #"idAbt" #"idScholar" # "idAbt"  #"idACM" # #gold standard columns
id2dfs = ["id2"] #["id2"] # ["idDBLP"]  #["idBuy"] #["idDBLP"] # ["idGoogleBase", "idDBLP", "id2"] #["idDBLP"] #["idGoogleBase", "idDBLP", "id2"] #  "idBuy" # "idDBLP" #"idBuy" #"idDBLP" #
num_candidates = 5
X_data = []
y_data = []
X_features = []


embedding_dim = 384 # Or df1['v'][0].shape[0] if 'v' contains numpy arrays
for file1, file2, file3, id1df, id2df in zip(files1, files2, files3, id1dfs, id2dfs):
  df2 = pd.read_parquet(f"./data/walmart_products_tuned.pqt")
  df2['id'] = pd.to_numeric(df2['id'], errors='coerce')
  df2.dropna(subset=['id'], inplace=True)
  df2['id'] = df2['id'].astype(int)

  df1 = pd.read_parquet(f"./data/amazon_products_tuned.pqt")
  df1['id'] = pd.to_numeric(df1['id'], errors='coerce')
  df1.dropna(subset=['id'], inplace=True)
  df1['id'] = df1['id'].astype(int)

  gold_standard = pd.read_csv(f"./data/truth_{file3}.tsv", sep="\t", encoding="utf-8", keep_default_na=False)
  gold_standard['id1'] = gold_standard['id1'].astype(int)
  gold_standard['id2'] = gold_standard['id2'].astype(int)

  #gold_standard['id1'] = pd.to_numeric(gold_standard['id1'])
  #gold_standard['id2'] = pd.to_numeric(gold_standard['id2'])
  minhash_titles1 = {row['id']: row['title_v'] for index, row in df1.iterrows()}
  minhash_titles2 = {row['id']: row['title_v'] for index, row in df2.iterrows()}
  minhash_models1 = {row['id']: row['modelno_v'] for index, row in df1.iterrows()}
  minhash_models2 = {row['id']: row['modelno_v'] for index, row in df2.iterrows()}
  minhash_categories1 = {row['id']: row['category_v'] for index, row in df1.iterrows()}
  minhash_categories2 = {row['id']: row['category_v'] for index, row in df2.iterrows()}
  minhash_brands1 = {row['id']: row['brand_v'] for index, row in df1.iterrows()}
  minhash_brands2 = {row['id']: row['brand_v'] for index, row in df2.iterrows()}

  titles1 = {row['id']: row['title'] for index, row in df1.iterrows()}
  titles2 = {row['id']: row['title'] for index, row in df2.iterrows()}
  prices1 = {row['id']: row['price'] for index, row in df1.iterrows()}
  prices2 = {row['id']: row['price'] for index, row in df2.iterrows()}
  categories1 = {row['id']: row['category'] for index, row in df1.iterrows()}
  categories2 = {row['id']: row['category'] for index, row in df2.iterrows()}
  brands1 = {row['id']: row['brand'] for index, row in df1.iterrows()}
  brands2 = {row['id']: row['brand'] for index, row in df2.iterrows()}
  models1 = {row['id']: row['modelno'] for index, row in df1.iterrows()}
  models2 = {row['id']: row['modelno'] for index, row in df2.iterrows()}

  vectors1 = df1['v'].tolist()
  vectors2 = df2['v'].tolist()
  d = 384
  faiss_db = faiss.IndexHNSWFlat(d, 32)
  faiss_db.hnsw.efConstruction = 60
  faiss_db.hnsw.efSearch = 16
  datav = np.array(vectors2).astype(np.float32)
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
      id_amazon = int(row[id1df])
      id_walmart = int(row[id2df])

      if not id_amazon in df1['id'].values or not id_walmart in df2['id'].values:
          print("Either of these is orphan", id_amazon, id_walmart)
          continue


      if id_amazon in embeddings1 and id_walmart in embeddings2:

        emb1 = embeddings1[id_amazon]
        emb2 = embeddings2[id_walmart]

        minhash_title1 = minhash_titles1[id_amazon]
        minhash_title2 = minhash_titles2[id_walmart]
        minhash_model1 = minhash_models1[id_amazon]
        minhash_model2 = minhash_models2[id_walmart]
        j_distance1 = jaccard(minhash_title1, minhash_title2)
        j_distance2 = jaccard(minhash_model1, minhash_model2)
        j_similarity1 = 1 - j_distance1
        j_similarity2 = 1 - j_distance2
        title1 = titles1[id_amazon]
        title2 = titles2[id_walmart]
        jw = jellyfish.jaro_winkler_similarity(str(title1), str(title2))
        category1 = categories1[id_amazon]
        category2 = categories2[id_walmart]
        jw2 = jellyfish.jaro_winkler_similarity(str(category1), str(category2))
        brand1 = brands1[id_amazon]
        brand2 = brands2[id_walmart]
        jw3 = jellyfish.jaro_winkler_similarity(str(brand1), str(brand2))
        model1 = models1[id_amazon]
        model2 = models2[id_walmart]
        model_sim = calculate_modelno_similarity(model1, model2)
        model_match = model_number_match(model1, model2)
        cat_jac = calculate_category_jaccard(category1, category2)
        tokens = count_differing_tokens(title1, title2)
        brand_sim = calculate_brand_similarity(brand1, brand2)
        brand_m = brand_match(brand1, brand2)
        price1 = prices1[id_amazon]
        price2 = prices2[id_walmart]
        price_diff = calculate_price_diff(price1, price2)
        # Ensure embeddings are numpy arrays and then concatenate
        combined_embedding = np.concatenate((np.array(emb1), np.array(emb2) ))
        #features = np.array([ j_similarity1,  model_sim, model_match, brand_sim, brand_m])
        features = np.array([j_similarity1, model_match, price_diff, jw])

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
                  minhash_model2 = minhash_models2[id_walmart_]
                  j_distance1 = jaccard(minhash_title1, minhash_title2)
                  j_distance2 = jaccard(minhash_model1, minhash_model2)
                  j_similarity1 = 1 - j_distance1
                  j_similarity2 = 1 - j_distance2
                  title2 = titles2[id_walmart_]
                  jw = jellyfish.jaro_winkler_similarity(str(title1), str(title2))
                  category2 = categories2[id_walmart_]
                  jw2 = jellyfish.jaro_winkler_similarity(str(category1), str(category2))
                  brand2 = brands2[id_walmart_]
                  jw3 = jellyfish.jaro_winkler_similarity(str(brand1), str(brand2))
                  model2 = models2[id_walmart_]
                  model_sim = calculate_modelno_similarity(model1, model2)
                  model_match = model_number_match(model1, model2)
                  cat_jac = calculate_category_jaccard(category1, category2)
                  tokens = count_differing_tokens(title1, title2)
                  brand_sim = calculate_brand_similarity(brand1, brand2)
                  brand_m = brand_match(brand1, brand2)
                  price2 = prices2[id_walmart_]
                  price_diff = calculate_price_diff(price1, price2)
                  combined_embedding = np.concatenate((np.array(emb1), np.array(emb2) ))
                  #features = np.array([ j_similarity1,  model_sim, model_match, brand_sim, brand_m])
                  features = np.array([j_similarity1, model_match, price_diff, jw])

                  X_data.append(combined_embedding)
                  X_features.append(features)
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

    if (random_id2, random_id1) not in gold_pairs_set and random_id1 in embeddings1 and  random_id2 in embeddings2:
        emb1 = embeddings1[random_id1]
        emb2 = embeddings2[random_id2]

        minhash_title1 = minhash_titles1[random_id1]
        minhash_title2 = minhash_titles2[random_id2]
        minhash_model1 = minhash_models1[random_id1]
        minhash_model2 = minhash_models2[random_id2]
        j_distance1 = jaccard(minhash_title1, minhash_title2)
        j_distance2 = jaccard(minhash_model1, minhash_model2)
        j_similarity1 = 1 - j_distance1
        j_similarity2 = 1 - j_distance2
        title1 = titles1[random_id1]
        title2 = titles2[random_id2]
        jw = jellyfish.jaro_winkler_similarity(str(title1), str(title2))
        category1 = categories1[random_id1]
        category2 = categories2[random_id2]
        jw2 = jellyfish.jaro_winkler_similarity(str(category1), str(category2))
        brand1 = brands1[random_id1]
        brand2 = brands2[random_id2]
        jw3 = jellyfish.jaro_winkler_similarity(str(brand1), str(brand2))
        model1 = models1[random_id1]
        model2 = models2[random_id2]
        price1 = prices1[random_id1]
        price2 = prices2[random_id2]
        price_diff = calculate_price_diff(price1, price2)
        model_sim = calculate_modelno_similarity(model1, model2)
        model_match = model_number_match(model1, model2)
        cat_jac = calculate_category_jaccard(category1, category2)
        tokens = count_differing_tokens(title1, title2)
        brand_sim = calculate_brand_similarity(brand1, brand2)
        brand_m = brand_match(brand1, brand2)

        combined_embedding = np.concatenate((np.array(emb1), np.array(emb2) ))
        #features = np.array([j_similarity1, model_sim, model_match, brand_sim, brand_m])
        features = np.array([j_similarity1,  model_match, price_diff, jw])

        X_data.append(combined_embedding)
        X_features.append(features)
        y_data.append(0) # Non-match
        generated_negatives += 1




  if not X_data:
    print("Error: No training data was generated. Check your data loading and pairing logic.")
    exit()

  print(f"Training data was generated. Ground truth matches: {sum(y_data)}, Ground truth non-matches: {len(y_data) - sum(y_data)}.")




X_data = np.array(X_data)
X_features = np.array(X_features)
y = np.array(y_data)

emb1_batch = X_data[:, :384]
emb2_batch = X_data[:, 384:]
numerator = np.einsum('ij,ij->i', emb1_batch, emb2_batch)
denominator = np.linalg.norm(emb1_batch, axis=1) * np.linalg.norm(emb2_batch, axis=1)
epsilon = 1e-7
cosine_sim_scores = (numerator / (denominator + epsilon)).reshape(-1, 1)
X_features = np.concatenate([X_features, cosine_sim_scores], axis=1)

diff_vectors = emb1_batch - emb2_batch
product_vectors = emb1_batch * emb2_batch
X_interactions = np.concatenate([diff_vectors, product_vectors], axis=1)


X_embeddings_train, X_embeddings_temp, \
X_interactions_train, X_interactions_temp, \
X_features_train, X_features_temp, \
y_train, y_temp = train_test_split(
    X_data, X_interactions, X_features, y,
    test_size=0.40, # Hold out 40% for val and test
    random_state=42,
    stratify=y
)

# --- Second split: Split the temporary set (40%) into validation (20%) and test (20%) ---
# We set test_size=0.5 to split the temp set (which is 40% of the total) in half.
# 50% of 40% is 20%.
X_embeddings_val, X_embeddings_test, \
X_interactions_val, X_interactions_test, \
X_features_val, X_features_test, \
y_val, y_test = train_test_split(
    X_embeddings_temp, X_interactions_temp, X_features_temp, y_temp,
    test_size=0.50, # Split the 40% into two 20% chunks
    random_state=42,
    stratify=y_temp # Stratify on the temporary labels
)



# Scale the engineered features (still a best practice)
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#X_eng_train_scaled = scaler.fit_transform(X_eng_train)
#X_eng_val_scaled = scaler.transform(X_eng_val)


#X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if np.sum(y) > 1 and np.sum(y) < len(y)-1 else None) # stratify if possible

embedding_input_shape = X_embeddings_train.shape[1]  # e.g., 768
features_input_shape = X_features_train.shape[1] # e.g., 6
interactions_input_shape = X_interactions.shape[1]

embedding_input = Input(shape=(embedding_input_shape,), name='embedding_input')
x1 = Dense(256, activation='relu')(embedding_input)
x1 = Dropout(0.2)(x1)
x1 = Dense(128, activation='relu')(x1)
# The output of this branch is the 'x1' tensor

interactions_input = Input(shape=(interactions_input_shape,), name='interactions_input')
x2 = Dense(256, activation='relu')(interactions_input)
x2 = Dropout(0.2)(x2)
x2 = Dense(128, activation='relu')(x2)

# --- Branch 2: Processes the Engineered Features ---
features_input = Input(shape=(features_input_shape,), name='features_input')
x3 = Dense(64, activation='relu')(features_input)
x3 = Dense(32, activation='relu')(x3)
# The output of this branch is the 'x2' tensor

# --- Combine the branches ---
combined = Concatenate()([x1, x2, x3])

# --- Add a final classifier head ---
final_dense = Dense(128, activation='relu')(combined)
final_dropout = Dropout(0.2)(final_dense)
output = Dense(1, activation='sigmoid', name='output')(final_dropout)

# The model takes a list of inputs and produces a single output
model = Model(inputs=[embedding_input, interactions_input, features_input], outputs=output)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# You can print a summary to visualize the architecture
model.summary()

# --- 4. TRAIN THE MODEL ---

# Note that the training data is now a LIST of two arrays
train_inputs = [X_embeddings_train, X_interactions_train, X_features_train]
val_inputs = [X_embeddings_val, X_interactions_val, X_features_val]
test_inputs = [X_embeddings_test, X_interactions_test, X_features_test]


# We will monitor validation loss. Training will stop if it doesn't improve for 10 epochs.
# The model will automatically be restored to the weights of the best epoch.
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1,
    mode="min",
    restore_best_weights=True
)
class_weight = {
    0: 1,  # The weight for the 'non-match' class
    1: 2.5   # The weight for the 'match' class
}
history = model.fit(
    train_inputs,
    y_train,
    validation_data=(val_inputs, y_val),
    epochs=100,
    batch_size=64,
    verbose=1,
    class_weight=class_weight,
    callbacks=[early_stopping_callback]
)

print("\nEvaluating on validation data...")
# The .predict() method also takes a list of inputs
y_pred_probs = model.predict(val_inputs)
y_pred = (y_pred_probs > 0.5).astype(int)

precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

print("\n--- Two-Branch Neural Network Results ---")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")


model_save_path = f"./data/er_{name}.keras"
model.save(model_save_path)
print(f"Model saved to {model_save_path}")
#exit()

from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
print(f"Validation rows: {y_val.shape}")
y_preds = model.predict(val_inputs)
precision, recall, thresholds = precision_recall_curve(y_val, y_preds)
epsilon = 1e-7
f1_scores = 2 * (precision * recall) / (precision + recall + epsilon)

# The last value of precision and recall are 1. and 0. respectively and do not have a corresponding threshold.
# We will ignore this last value for plotting and finding the best threshold.
f1_scores = f1_scores[:-1]
precision = precision[:-1]
recall = recall[:-1]


# --- 4. FIND THE BEST THRESHOLD based on the highest F1-score ---

# Find the index of the best F1 score
best_f1_idx = np.argmax(f1_scores)

# Get the best threshold, precision, recall, and f1-score
best_threshold = thresholds[best_f1_idx]
best_precision = precision[best_f1_idx]
best_recall = recall[best_f1_idx]
best_f1_score = f1_scores[best_f1_idx]

print(f"Best Threshold: {best_threshold:.4f}")
print(f"  - Precision at best threshold: {best_precision:.4f}")
print(f"  - Recall at best threshold: {best_recall:.4f}")
print(f"  - F1-Score at best threshold: {best_f1_score:.4f}")


# --- 5. PLOT THE PRECISION-RECALL CURVE ---

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))

# Plot the main P-R curve
plt.plot(recall, precision, marker='.', label='Precision-Recall Curve', zorder=2)

# Plot the point for the best threshold
plt.scatter(best_recall, best_precision, marker='o', color='red', s=100,
            label=f'Best Threshold ({best_threshold:.2f})', zorder=3)

# Annotate the best point
plt.annotate(f'best precision={best_precision:.2f},\nbest recall={best_recall:.2f},\nF1={best_f1_score:.2f}',
             xy=(best_recall, best_precision),
             xytext=(best_recall + 0.05, best_precision - 0.1),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))

# Formatting
plt.title('Precision-Recall Curve', fontsize=16)
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.legend(loc='lower left')
plt.grid(True)
plt.show()

print("\nEvaluating on test data...")
# The .predict() method also takes a list of inputs
y_pred_probs = model.predict(test_inputs)
y_pred = (y_pred_probs > best_threshold).astype(int)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n--- Two-Branch Neural Network Results ---")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")



