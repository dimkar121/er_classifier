import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# Ensure tensorflow is installed if you are running this locally
# For Google Colab or similar environments, Keras is usually pre-installed.
# If not, you might need: pip install tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,  Dense, Dropout, Concatenate, Subtract, Multiply
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import math
import faiss
# --- 1. Assume df1, df2, and gold_standard are loaded ---
# df1: Parquet DataFrame with 'id' and 'v'
# df2: Parquet DataFrame with 'id' and 'v'
# gold_standard: DataFrame with ids

# --- Example: Data Preparation (Simplified) ---
# This is a conceptual illustration. You'll need to implement the actual lookup
# and pairing logic robustly.
name="general"
files1 =  ["Scholar","votersA", "fodors"]
files2 =  ["DBLP2", "votersB", "zagats" ]
files3 =  ["Scholar_DBLP", "voters", "fodors_zagats"]
id1dfs =  ["idScholar", "id1","idFodors"] # #gold standard columns
id2dfs =  ["idDBLP","id2", "idZagats"]
num_candidates = 5
X_data = []
y_data = []
embedding_dim = 384 # Or df1['v'][0].shape[0] if 'v' contains numpy arrays
for file1, file2, file3, id1df, id2df in zip(files1, files2, files3, id1dfs, id2dfs):
  df1 = pd.read_parquet(f"./data/{file1}_embedded_mini.pqt")
  df2 = pd.read_parquet(f"./data/{file2}_embedded_mini.pqt")
  gold_standard = pd.read_csv(f"./data/truth_{file3}.csv", sep=",", encoding="utf-8", keep_default_na=False)

  vectors1 = df1['v'].tolist()
  vectors2 = df2['v'].tolist()
  d = 384
  faiss_db = faiss.IndexHNSWFlat(d, 32)
  faiss_db.hnsw.efConstruction = 64
  faiss_db.hnsw.efSearch = 16
  datav = np.array(vectors2).astype(np.float32)
  faiss_db.add(datav)
  ids2_ = df2['id'].tolist()

  # Let's assume MiniLM embeddings have a dimension of 384

  # Create dictionaries for quick embedding lookups
  embeddings1 = {row['id']: row['v'] for index, row in df1.iterrows()}
  embeddings2 = {row['id']: row['v'] for index, row in df2.iterrows()}

  num_samples_to_take = math.ceil(len(gold_standard) * 0.50)
  print(f"{num_samples_to_take} pairs will be sampled.")
  sampled_gold_standard = gold_standard.sample(n=num_samples_to_take, random_state=42, replace=False)
  # Positive Pairs
  for index, row in sampled_gold_standard.iterrows():
      id1 = row[id1df]
      id2 = row[id2df]

      if id1 in embeddings1 and id2 in embeddings2:
        emb1 = embeddings1[id1]
        emb2 = embeddings2[id2]
        # Ensure embeddings are numpy arrays and then concatenate
        combined_embedding = np.concatenate((np.array(emb1), np.array(emb2)))
        X_data.append(combined_embedding)
        y_data.append(1) # Match

        distances, indices1 = faiss_db.search(np.array([emb1]), num_candidates)  # return scholars
        for ind, ds in zip(indices1[0], distances[0]):
              id2_ =ids2_[ind]
              if id2_ != id2:
                  emb2 = embeddings2[id2_]
                  combined_embedding = np.concatenate((np.array(emb1), np.array(emb2)))
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
        combined_embedding = np.concatenate((np.array(emb1), np.array(emb2)))
        X_data.append(combined_embedding)
        y_data.append(0) # Non-match
        generated_negatives += 1





print(f"Training data was generated. Ground truth matches: {sum(y_data)}, Ground truth non-matches: {len(y_data) - sum(y_data)}.")

X = np.array(X_data)
y = np.array(y_data)


print(f"X-data {X.shape} y_data {y.shape}")


emb1_batch = X[:, :384]
emb2_batch = X[:, 384:]
diff_vectors = emb1_batch - emb2_batch
product_vectors = emb1_batch * emb2_batch
X_interactions = np.concatenate([diff_vectors, product_vectors], axis=1)



# Split data
X_embeddings_train, X_embeddings_val, X_interactions_train, X_interactions_val, y_train, y_val = (
    train_test_split(X, X_interactions, y, test_size=0.2, random_state=42, stratify=y)) # stratify if possible


embedding_input_shape = X_embeddings_train.shape[1]  # e.g., 768
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

combined = Concatenate()([x1, x2])

# --- Add a final classifier head ---
final_dense = Dense(128, activation='relu')(combined)
final_dropout = Dropout(0.2)(final_dense)
output = Dense(1, activation='sigmoid', name='output')(final_dropout)

# The model takes a list of inputs and produces a single output
model = Model(inputs=[embedding_input, interactions_input], outputs=output)


# --- 3. Compile the Model ---
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

train_inputs = [X_embeddings_train, X_interactions_train ]
val_inputs = [X_embeddings_val, X_interactions_val]


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

exit()







# --- Making Predictions ---
file1 =  "ACM" #  "fodors" #"ACM" # "Abt" # "Amazon" #  "ACM" #"Scholar" "Abt"
file2 =  "DBLP"  #"zagats" #"DBLP" # "Buy" #"Google" #"DBLP" #"DBLP2" "Buy"
file3 =  "ACM_DBLP" #"fodors_zagats" # #  "Amazon_googleProducts" #"ACM_DBLP" #"Scholar_DBLP" "abt_buy"
id1df =  "idACM"  #"idFodors" #  # "idAmazon" # "idACM" #"idScholar" #gold standard columns "idAbt"
id2df =  "idDBLP"  #"idZagats" # #"idGoogleBase" # "idDBLP" #"idDBLP" "idBuy"

df1 = pd.read_parquet(f"./data/{file1}_embedded_mini.pqt")
df2 = pd.read_parquet(f"./data/{file2}_embedded_mini.pqt")
gold_standard = pd.read_csv(f"./data/truth_{file3}.csv", sep=",", encoding="utf-8", keep_default_na=False)
embeddings1 = {row['id']: row['v'] for index, row in df1.iterrows()}
embeddings2 = {row['id']: row['v'] for index, row in df2.iterrows()}
all_ids_1 = list(embeddings1.keys())
all_ids_2 = list(embeddings2.keys())



X_test_fixed = []
y_test_fixed_true = []

# --- 1. Select 500 "Match" Pairs from Gold Standard ---
num_matches_to_select = 10_000

if len(gold_standard) < num_matches_to_select:
    print(f"Warning: Gold standard has only {len(gold_standard)} pairs. "
          f"Selecting all of them ({len(gold_standard)}) as matches.")
    num_matches_to_select = len(gold_standard)
    if num_matches_to_select == 0:
        print("Error: Gold standard is empty. Cannot select matches.")
        exit()  # Or handle as appropriate

# Sample from gold_standard without replacement if possible, or with replacement if needed (though less ideal)
# For this fixed set, it's better to ensure unique pairs if len(gold_standard) >= num_matches_to_select
try:
    match_samples_df = gold_standard.sample(n=num_matches_to_select, random_state=42, replace=False)
except ValueError as e:
    if 'Cannot take a sample larger than the population when replace=False' in str(e):
        print(f"Warning: Cannot sample {num_matches_to_select} unique matches. "
              f"Sampling with replacement or taking all available unique matches ({len(gold_standard)}).")
        if len(gold_standard) < num_matches_to_select:
            match_samples_df = gold_standard.sample(n=num_matches_to_select, random_state=42, replace=True)
        else:  # Should not happen if initial check passed, but as a fallback
            match_samples_df = gold_standard.sample(n=len(gold_standard), random_state=42, replace=False)

    else:
        raise e  # Re-raise other ValueErrors

print(f"Selected {len(match_samples_df)} matches from the gold standard.")

for index, row in match_samples_df.iterrows():
    id1 = row[id1df]
    id2 = row[id2df]

    if id1 in embeddings1 and id2 in embeddings2:
        emb1 = embeddings1[id1]
        emb2 = embeddings2[id2]
        # Ensure embeddings are numpy arrays before concatenation
        combined_embedding = np.concatenate((np.array(emb1), np.array(emb2)))
        X_test_fixed.append(combined_embedding)
        y_test_fixed_true.append(1)  # Label for match
    else:
        print(f"Warning: Could not find embeddings for gold standard pair: "
              f"ID1 {id1}, ID2 {id2}. Skipping this pair.")

actual_matches_added = sum(1 for label in y_test_fixed_true if label == 1)
print(f"Successfully added {actual_matches_added} match pairs to the fixed test set.")

# --- 2. Generate 500 "Non-Match" Pairs ---
num_non_matches_to_generate = 500
generated_non_matches_count = 0
max_attempts_non_match = num_non_matches_to_generate * 100  # Safety break
current_attempts = 0

if not all_ids_1 or not all_ids_2:
    print("ERROR: lists are empty. Cannot generate non-matches.")
    exit()

print(f"Generating {num_non_matches_to_generate} non-matches.")
while generated_non_matches_count < num_non_matches_to_generate and current_attempts < max_attempts_non_match:
    current_attempts += 1
    random_id1 = np.random.choice(all_ids_1)
    random_id2 = np.random.choice(all_ids_2)

    # Check if this randomly generated pair is NOT in the gold standard
    if (random_id2, random_id1) not in gold_pairs_set:
        if random_id1 in embeddings1 and random_id2 in embeddings2:
            emb1 = embeddings1[random_id1]
            emb2 = embeddings2[random_id2]
            combined_embedding = np.concatenate((np.array(emb1), np.array(emb2)))

            X_test_fixed.append(combined_embedding)
            y_test_fixed_true.append(0)  # Label for non-match
            generated_non_matches_count += 1
        # else: # This case should be rare if all_ids1/all_ids2 are from keys of embeddings dicts

    if current_attempts % (max_attempts_non_match // 10) == 0 and current_attempts > 0:  # Log progress
        print(f"  Attempts: {current_attempts}, Non-matches generated: {generated_non_matches_count}")

if generated_non_matches_count < num_non_matches_to_generate:
    print(f"Warning: Could only generate {generated_non_matches_count} non-match pairs "
          f"after {max_attempts_non_match} attempts. The test set will have fewer non-matches than desired.")
else:
    print(f"Successfully generated {generated_non_matches_count} non-match pairs.")

# --- 3. Finalize and Shuffle ---
X_test_fixed = np.array(X_test_fixed)
y_test_fixed_true = np.array(y_test_fixed_true)
from sklearn.utils import shuffle
if X_test_fixed.shape[0] > 0:
    X_test_fixed, y_test_fixed_true = shuffle(X_test_fixed, y_test_fixed_true, random_state=42)
    print(f"\nFixed test set created with {X_test_fixed.shape[0]} total samples.")
    print(f"  Number of matches: {np.sum(y_test_fixed_true == 1)}")
    print(f"  Number of non-matches: {np.sum(y_test_fixed_true == 0)}")
    print(f"  Shape of X_test_fixed: {X_test_fixed.shape}")
    print(f"  Shape of y_test_fixed_true: {y_test_fixed_true.shape}")
else:
    print("Error: No data was added to the fixed test set. Check previous warnings/errors.")



# --- 7. Make Predictions on the Random Pairs ---
print("\n--- Making Predictions on Random Pairs ---")

predictions_prob_random = model.predict(X_test_fixed)
predictions_class_random = (predictions_prob_random > 0.5).astype(int).flatten() # Flatten if it's a column vector

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report # For evaluation
# --- 8. Evaluate the Model on these Random Pairs ---
print("\n--- Evaluating Model on Randomly Generated Pairs ---")
if len(y_test_fixed_true) > 0:
    accuracy_random = accuracy_score(y_test_fixed_true, predictions_class_random)
    precision_random = precision_score(y_test_fixed_true, predictions_class_random, zero_division=0)
    recall_random = recall_score(y_test_fixed_true, predictions_class_random, zero_division=0)
    f1_random = f1_score(y_test_fixed_true, predictions_class_random, zero_division=0)
    conf_matrix_random = confusion_matrix(y_test_fixed_true, predictions_class_random)

    print(f"Accuracy on random pairs: {accuracy_random:.4f}")
    print(f"Precision on random pairs: {precision_random:.4f}")
    print(f"Recall on random pairs: {recall_random:.4f}")
    print(f"F1-score on random pairs: {f1_random:.4f}")
    print("Confusion Matrix on random pairs:")
    print(conf_matrix_random)
    print("\nClassification Report on random pairs:")
    print(classification_report(y_test_fixed_true, predictions_class_random, zero_division=0))

    #You can also look at the distribution of prediction probabilities
    import matplotlib.pyplot as plt
    plt.hist(predictions_prob_random[y_test_fixed_true == 0], bins=50, alpha=0.5, label='True Non-Matches')
    plt.hist(predictions_prob_random[y_test_fixed_true == 1], bins=50, alpha=0.5, label='True Matches')
    plt.xlabel("Predicted Probability of Match")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Distribution of Prediction Probabilities on Random Test Set")
    plt.show()
else:
    print("No random pairs were generated or no true labels available for evaluation.")

# Example of how to predict for a specific new pair (if you have their embeddings)
# if 's_new_id' in scholar_embeddings and 'd_new_id' in dblp_embeddings:
#     emb_s_new = scholar_embeddings['s_new_id']
#     emb_d_new = dblp_embeddings['d_new_id']
#     new_pair_concatenated = np.concatenate((np.array(emb_s_new), np.array(emb_d_new))).reshape(1, -1)
#     prediction_single_pair = model.predict(new_pair_concatenated)
#     print(f"\nPrediction for a specific new pair (s_new_id, d_new_id): {prediction_single_pair[0][0]:.4f}")
#     if prediction_single_pair[0][0] > 0.5:
#         print("Predicted as: Match")
#     else:
#         print("Predicted as: Non-Match")