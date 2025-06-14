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
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.metrics.pairwise import cosine_similarity
# --- 1. Assume df1, df2, and gold_standard are loaded ---
# df1: Parquet DataFrame with 'id' and 'v'
# df2: Parquet DataFrame with 'id' and 'v'
# gold_standard: DataFrame with ids

# --- Example: Data Preparation (Simplified) ---
# This is a conceptual illustration. You'll need to implement the actual lookup
# and pairing logic robustly.
name="wdc"
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


def prepare_data_from_splits(df_a, df_b, df_split):
    """
    Takes a split dataframe (train, valid, or test) and prepares the feature
    and label matrices for the 3-branch neural network.

    Args:
        df_a (pd.DataFrame): The full Table A dataframe with embeddings.
        df_b (pd.DataFrame): The full Table B dataframe with embeddings.
        df_split (pd.DataFrame): The dataframe for the specific split (e.g., train.csv).

    Returns:
        tuple: A tuple containing (X1_embeddings, X2_interactions, X3_engineered, y_labels).
    """
    print(f"Processing split with {len(df_split)} pairs...")

    # --- Merge to create a full pairs dataframe for this split ---
    pairs = df_split.copy()
    pairs = pd.merge(pairs, df_a, left_on='ltable_id', right_on='id', suffixes=('_a', '_b'))
    pairs = pd.merge(pairs, df_b, left_on='rtable_id', right_on='id', suffixes=('_a', '_b'))

    # --- Prepare Engineered Features (Input 3) ---
    print("  - Engineering handcrafted features...")
    features = pd.DataFrame()
    features['name_jaccard_sim'] = pairs.apply(lambda r: jaccard(r['mv1_a'], r['mv1_b']), axis=1)
    features['desc_jaccard_sim'] = pairs.apply(lambda r: jaccard(r['mv2_a'], r['mv2_b']),  axis=1)
    pairs['models_a'] = pairs['title_a'].apply(extract_model)
    pairs['models_b'] = pairs['title_b'].apply(extract_model)
    features['models_match'] = pairs.apply(lambda r: are_models_matching(r['models_a'], r['models_b']), axis=1)
    features['price_match'] = pairs.apply(lambda r: calculate_price_diff(r['price_a'], r['price_b']), axis=1)
    features['title_jaro_sim'] = pairs.apply(lambda r: jellyfish.jaro_winkler_similarity(r['title_a'], r['title_b']), axis=1)

    # --- Prepare Embedding and Interaction Features (Inputs 1 & 2) ---
    print("  - Preparing embedding and interaction features...")
    emb1_batch = np.array(pairs['v_a'].tolist())
    emb2_batch = np.array(pairs['v_b'].tolist())

    numerator = np.einsum('ij,ij->i', emb1_batch, emb2_batch)
    denominator = np.linalg.norm(emb1_batch, axis=1) * np.linalg.norm(emb2_batch, axis=1)
    epsilon = 1e-7  # To prevent division by zero for zero-vectors
    features['cosine_sim'] = numerator / (denominator + epsilon)
    # Model number matching requires pre-extracting the sets

    X3_engineered = features.to_numpy()

    # Input 1: Concatenated Embeddings
    X1_embeddings = np.concatenate([emb1_batch, emb2_batch], axis=1)

    # Input 2: Interaction Vectors
    diff_vectors = emb1_batch - emb2_batch
    product_vectors = emb1_batch * emb2_batch
    X2_interactions = np.concatenate([diff_vectors, product_vectors], axis=1)

    # --- Prepare Labels ---
    y_labels = pairs['label'].to_numpy()

    return X1_embeddings, X2_interactions, X3_engineered, y_labels



embedding_dim = 384 # Or df1['v'][0].shape[0] if 'v' contains numpy arrays
df1 = pd.read_parquet(f"./data/wdc/tableA_.pqt")
df2 = pd.read_parquet(f"./data/wdc/tableB_.pqt")
df1['id'] = pd.to_numeric(df1['id'])
df2['id'] = pd.to_numeric(df2['id'])

#gold_standard = pd.read_csv(f"./data/wdc/gold_standard.csv", sep=",", encoding="utf-8", keep_default_na=False)
minhash_names1 = {row['id']: row['mv1'] for index, row in df1.iterrows()}
minhash_names2 = {row['id']: row['mv1'] for index, row in df2.iterrows()}
minhash_descrs1 = {row['id']: row['mv2'] for index, row in df1.iterrows()}
minhash_descrs2 = {row['id']: row['mv2'] for index, row in df2.iterrows()}
prices1 = {row['id']: row['price'] for index, row in df1.iterrows()}
prices2 = {row['id']: row['price'] for index, row in df2.iterrows()}
names1 = {row['id']: row['title'] for index, row in df1.iterrows()}
names2 = {row['id']: row['title'] for index, row in df2.iterrows()}
df1['models'] = df1['title'].apply(extract_model)
df2['models'] = df2['title'].apply(extract_model)
models1 = {row['id']: row['models'] for index, row in df1.iterrows()}
models2 = {row['id']: row['models'] for index, row in df2.iterrows()}
#all_brands = df2_minhash['brand'].dropna().unique()
#brands_list = sorted([str(b).lower() for b in all_brands if len(str(b)) > 2], key=len, reverse=True)
#df1_minhash['brand'] = df1_minhash['name'].apply(lambda text: find_brand_in_text(text, brands_list))
brands1 = {row['id']: row['brand'] for index, row in df1.iterrows()}
brands2 = {row['id']: row['brand'] for index, row in df2.iterrows()}
vectors1 = df1['v'].tolist()
vectors2 = df2['v'].tolist()
ids1_ = df1['id'].tolist()

# Create dictionaries for quick embedding lookups
embeddings1 = {row['id']: row['v'] for index, row in df1.iterrows()}
embeddings2 = {row['id']: row['v'] for index, row in df2.iterrows()}


df_train = pd.read_csv('./data/wdc/train.csv')
df_test = pd.read_csv('./data/wdc/test.csv')
df_valid = pd.read_csv('./data/wdc/valid.csv')
X1_train, X2_train, X3_train, y_train = prepare_data_from_splits(df1, df2, df_train)
X1_val, X2_val, X3_val, y_val = prepare_data_from_splits(df1, df2, df_valid)
X1_test, X2_test, X3_test, y_test = prepare_data_from_splits(df1, df2, df_test)

print("\n--- Final Data Shapes ---")
print(f"Train shapes: X1={X1_train.shape}, X2={X2_train.shape}, X3={X3_train.shape}, y={y_train.shape}")
print(f"Valid shapes: X1={X1_val.shape}, X2={X2_val.shape}, X3={X3_val.shape}, y={y_val.shape}")
print(f"Test shapes:  X1={X1_test.shape}, X2={X2_test.shape}, X3={X3_test.shape}, y={y_test.shape}")

d=768
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



#X_embeddings_train, X_embeddings_val, X_interactions_train, X_interactions_val,  X_features_train, X_features_val, y_train,  y_val = train_test_split(
 #   X_data, X_interactions, X_features, y, test_size=0.20, random_state=42, stratify=y)


X_embeddings_train = X1_train
X_embeddings_val = X1_val
X_interactions_train = X2_train
X_interactions_val = X1_val
X_features_train =  X3_train
X_features_val = X3_val
y_train = y_train
y_val = y_val


# Scale the engineered features (still a best practice)
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#X_eng_train_scaled = scaler.fit_transform(X_eng_train)
#X_eng_val_scaled = scaler.transform(X_eng_val)



embedding_input_shape = X_embeddings_train.shape[1]  # e.g., 768
features_input_shape = X_features_train.shape[1] # e.g., 6
interactions_input_shape = X_interactions.shape[1]

embedding_input = Input(shape=(embedding_input_shape,), name='embedding_input')
x1 = Dense(768, activation='relu')(embedding_input)
x1 = Dropout(0.2)(x1)
x1 = Dense(384, activation='relu')(x1)
# The output of this branch is the 'x1' tensor

#x1 = Dense(512, activation='relu')(embedding_input) # Start with a wider layer
#x1 = Dropout(0.3)(x1)
#x1 = Dense(256, activation='relu')(x1)
#x1 = Dropout(0.2)(x1)
#x1 = Dense(128, activation='relu')(x1) # Add the extra layer

interactions_input = Input(shape=(interactions_input_shape,), name='interactions_input')
x2 = Dense(768, activation='relu')(interactions_input)
x2 = Dropout(0.2)(x2)
x2 = Dense(384, activation='relu')(x2)

#x2 = Dense(512, activation='relu')(interactions_input) # Start with a wider layer
#x2 = Dropout(0.3)(x2)
#x2 = Dense(256, activation='relu')(x2)
#x2 = Dropout(0.2)(x2)
#x2 = Dense(128, activation='relu')(x2) # Add the extra layer




# --- Branch 2: Processes the Engineered Features ---
features_input = Input(shape=(features_input_shape,), name='features_input')
x3 = Dense(64, activation='relu')(features_input)
x3 = Dense(32, activation='relu')(x3)
# The output of this branch is the 'x2' tensor

# --- Combine the branches ---
combined = Concatenate()([x1, x2, x3])

# --- Add a final classifier head ---
final_dense = Dense(256, activation='relu')(combined)
final_dropout = Dropout(0.2)(final_dense)
final_dense = Dense(128, activation='relu')(final_dense)
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
test_inputs = [X1_test, X2_test, X3_test]

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

print("\n--- Three-Branch Neural Network Results ---")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")


model_save_path = f"./data/wdc/er_wdc.keras"
model.save(model_save_path)
print(f"Model saved to {model_save_path}")



from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
print(f"Validation rows: {y_val.shape}")


y_preds = model.predict(test_inputs)
precision, recall, thresholds = precision_recall_curve(y_test, y_preds)
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





