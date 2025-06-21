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
import jellyfish
from sklearn.metrics import precision_score, recall_score, f1_score
# --- 1. Assume df1, df2, and gold_standard are loaded ---
# df1: Parquet DataFrame with 'id' and 'v'
# df2: Parquet DataFrame with 'id' and 'v'
# gold_standard: DataFrame with ids

# --- Example: Data Preparation (Simplified) ---
# This is a conceptual illustration. You'll need to implement the actual lookup
# and pairing logic robustly.
name="amazon_google"
files1 =["Amazon"] #["votersA"] # ["Scholar"] #["Abt"] #["Scholar"] #["Amazon", "Scholar", "votersA"] #["Scholar"] # ["Amazon", "Scholar", "votersA"]   #"Abt" # "Scholar"  #"Abt" # "ACM" #
files2 = ["Google"] #["votersB"] #["DBLP2"] # ["Buy"] #["DBLP2"] #["Google", "DBLP2" , "votersB"] #["DBLP2"] #["Google", "DBLP2" , "votersB"]  # "Buy" #"DBLP2" # #"DBLP" #
files3 =["Amazon_googleProducts"] # ["voters"] #["Scholar_DBLP"] # ["abt_buy"] #["Scholar_DBLP"] #["Amazon_googleProducts", "Scholar_DBLP", "voters"] #["Scholar_DBLP"] #["Amazon_googleProducts", "Scholar_DBLP", "voters"]# "abt_buy" #"Scholar_DBLP" # # "ACM_DBLP" #
id1dfs = ["idAmazon"] #["id1"] # ["idScholar"] # ["idAbt"] #["idScholar"] # ["idAmazon", "idScholar", "id1"] #["idScholar"] #["idAmazon", "idScholar", "id1"] #"idAbt" #"idScholar" # "idAbt"  #"idACM" # #gold standard columns
id2dfs = ["idGoogleBase"] #["id2"] # ["idDBLP"]  #["idBuy"] #["idDBLP"] # ["idGoogleBase", "idDBLP", "id2"] #["idDBLP"] #["idGoogleBase", "idDBLP", "id2"] #  "idBuy" # "idDBLP" #"idBuy" #"idDBLP" #
num_candidates = 5
X_data = []
y_data = []
X_features = []




def minhash(s, num_hashes=120):
      if s is None or len(s) < 2:
          s="  ";
      set_elements = {s[i:i+2] for i in range(len(s)-1)}
      signature = np.zeros(num_hashes, dtype=np.uint8)
      for i in range(num_hashes):
          min_hash = min(mmh3.hash(str(el), seed=i) & 0xFFFFFFFF for el in set_elements)  # 32-bit unsigned
          signature[i] = 1 if (min_hash % 2) == 0 else 0  # Convert to binary (0 or 1)
      return signature



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


embedding_dim = 384 # Or df1['v'][0].shape[0] if 'v' contains numpy arrays
for file1, file2, file3, id1df, id2df in zip(files1, files2, files3, id1dfs, id2dfs):
  df1 = pd.read_parquet(f"./data/Amazon_embedded_mini_ft.pqt")
  df2 = pd.read_parquet(f"./data/Google_embedded_mini_ft.pqt")
  gold_standard = pd.read_csv(f"./data/truth_{file3}.csv", sep=",", encoding="utf-8", keep_default_na=False)


  df1_minhash = pd.read_parquet(f"./data/Amazon_embedded_minhash_all.pqt")
  df2_minhash = pd.read_parquet(f"./data/Google_embedded_minhash_all.pqt")

  #df1_minhash["namev"] = df1_minhash.apply(lambda row: minhash(row["title"]), axis=1)
  #df2_minhash["namev"] = df2_minhash.apply(lambda row: minhash(row["name"]), axis=1)
  #df1_minhash["descriptionv"] = df1_minhash.apply(lambda row: minhash(row["description"]), axis=1)
  #df2_minhash["descriptionv"] = df2_minhash.apply(lambda row: minhash(row["description"]), axis=1)
  #df1_minhash.to_parquet('./data/Amazon_embedded_minhash_all.pqt', engine='pyarrow')
  #df2_minhash.to_parquet('./data/Google_embedded_minhash_all.pqt', engine='pyarrow')
  #exit()

  minhash_names1 = {row['id']: row['namev'] for index, row in df1_minhash.iterrows()}
  minhash_names2 = {row['id']: row['namev'] for index, row in df2_minhash.iterrows()}
  minhash_descrs1 = {row['id']: row['descriptionv'] for index, row in df1_minhash.iterrows()}
  minhash_descrs2 = {row['id']: row['descriptionv'] for index, row in df2_minhash.iterrows()}
  df1['models'] = df1['title'].apply(extract_model)
  df2['models'] = df2['name'].apply(extract_model)

  models1 = {row['id']: row['models'] for index, row in df1.iterrows()}
  models2 = {row['id']: row['models'] for index, row in df2.iterrows()}
  names1 = {row['id']: row['title'] for index, row in df1.iterrows()}
  names2 = {row['id']: row['name'] for index, row in df2.iterrows()}
  prices1 = {row['id']: row['price'] for index, row in df1.iterrows()}
  prices2 = {row['id']: row['price'] for index, row in df2.iterrows()}
  all_brands = df1['manufacturer'].dropna().unique()
  brands_list = sorted([str(b).lower() for b in all_brands if len(str(b)) > 2], key=len, reverse=True)
  #df1_minhash['brand'] = df1_minhash['name'].apply(lambda text: find_brand_in_text(text, brands_list))
  brands1 = {row['id']: row['manufacturer'] for index, row in df1.iterrows()}
  df2['brand'] = df2_minhash['name'].apply(lambda text: find_brand_in_text(text, brands_list))
  brands2 = {row['id']: row['brand'] for index, row in df2.iterrows()}

  vectors1 = df1['v'].tolist()
  vectors2 = df2['v'].tolist()
  d = 384
  faiss_db = faiss.IndexHNSWFlat(d, 32)
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
      id1 = row[id1df]
      id2 = row[id2df]

      if id1 in embeddings1 and id2 in embeddings2:
        emb1 = embeddings1[id1]
        emb2 = embeddings2[id2]


        minhash_name1 = minhash_names1[id1]
        minhash_name2 = minhash_names2[id2]
        minhash_descr1 = minhash_descrs1[id1]
        minhash_descr2 = minhash_descrs2[id2]

        j_distance1 = jaccard(minhash_name1, minhash_name2)
        j_distance2 = jaccard(minhash_descr1, minhash_descr2)
        j_similarity1 = 1 - j_distance1
        j_similarity2 = 1 - j_distance2
        model1 = models1[id1]
        model2 = models2[id2]
        name1 = names1[id1]
        name2 = names2[id2]
        price1 = prices1[id1]
        price2 = prices2[id2]
        brand1 = brands1[id1]
        brand2 = brands2[id2]
        br = check_brand_match(brand1, brand2)
        jw = jellyfish.jaro_winkler_similarity(str(name1), str(name2))
        price_diff = calculate_price_diff(price1, price2)
        m = are_models_matching(model1, model2)
        # Ensure embeddings are numpy arrays and then concatenate

        combined_embedding = np.concatenate((np.array(emb1), np.array(emb2) ))
        features = np.array([j_similarity1, j_similarity2, m, price_diff, jw])

        X_data.append(combined_embedding)
        X_features.append(features)
        y_data.append(1) # Match

        distances, indices1 = faiss_db.search(np.array([emb2]), num_candidates)
        for ind, ds in zip(indices1[0], distances[0]):
              id1_ =ids1_[ind]
              if id1_ != id1:
                  emb1 = embeddings1[id1_]

                  minhash_name1 = minhash_names1[id1_]
                  minhash_descr1 = minhash_descrs1[id1_]
                  model1 = models1[id1_]
                  name1 = names1[id1_]
                  price1 = prices1[id1_]
                  brand1 = brands1[id1_]
                  br = check_brand_match(brand1, brand2)
                  j_distance1 = jaccard(minhash_name1, minhash_name2)
                  j_distance2 = jaccard(minhash_descr1, minhash_descr2)
                  j_similarity1 = 1 - j_distance1
                  j_similarity2 = 1 - j_distance2
                  m = are_models_matching(model1, model2)
                  price_diff = calculate_price_diff(price1, price2)
                  jw = jellyfish.jaro_winkler_similarity(str(name1), str(name2))

                  combined_embedding = np.concatenate((np.array(emb1), np.array(emb2) ))
                  features = np.array([j_similarity1, j_similarity2, m, price_diff, jw])
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

    if (random_id2, random_id1) not in gold_pairs_set:
        emb1 = embeddings1[random_id1]
        emb2 = embeddings2[random_id2]

        minhash_name1 = minhash_names1[random_id1]
        minhash_name2 = minhash_names2[random_id2]
        minhash_descr1 = minhash_descrs1[random_id1]
        minhash_descr2 = minhash_descrs2[random_id2]
        model1 = models1[random_id1]
        model2 = models2[random_id2]
        m = are_models_matching(model1, model2)
        name1 = names1[random_id1]
        name2 = names2[random_id2]
        price1 = prices1[random_id1]
        price2 = prices2[random_id2]
        brand1 = brands1[random_id1]
        brand2 = brands2[random_id2]
        br = check_brand_match(brand1, brand2)
        jw = jellyfish.jaro_winkler_similarity(str(name1), str(name2))
        price_diff = calculate_price_diff(price1, price2)
        j_distance1 = jaccard(minhash_name1, minhash_name2)
        j_distance2 = jaccard(minhash_descr1, minhash_descr2)
        j_similarity1 = 1 - j_distance1
        j_similarity2 = 1 - j_distance2

        combined_embedding = np.concatenate((np.array(emb1), np.array(emb2) ))
        features = np.array([j_similarity1,j_similarity2, m, price_diff, jw])
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


X_embeddings_train, X_embeddings_val, X_interactions_train, X_interactions_val,  X_features_train, X_features_val, y_train,  y_val = train_test_split(
    X_data, X_interactions, X_features, y, test_size=0.20, random_state=42, stratify=y)


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
#combined = Concatenate()([x1, x2, x3])
combined = Concatenate()([x1, x2])
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

train_inputs = [X_embeddings_train, X_interactions_train, X_features_train]
val_inputs = [X_embeddings_val, X_interactions_val, X_features_val]


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





