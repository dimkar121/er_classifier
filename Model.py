import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, Subtract, Multiply

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import precision_score, recall_score, f1_score



def train(name, X_data, X_features, X_interactions, y):
    X_embeddings_train, X_embeddings_temp, \
        X_interactions_train, X_interactions_temp, \
        X_features_train, X_features_temp, \
        y_train, y_temp = train_test_split(
        X_data, X_interactions, X_features, y,
        test_size=0.40,  # Hold out 40% for val and test
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
        test_size=0.50,  # Split the 40% into two 20% chunks
        random_state=42,
        stratify=y_temp  # Stratify on the temporary labels
    )

    embedding_input_shape = X_embeddings_train.shape[1]  # e.g., 768
    features_input_shape = X_features_train.shape[1]  # e.g., 6
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
        1: 2.5  # The weight for the 'match' class
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

    model_save_path = f"./data/er_{name}.keras"
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    # exit()

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

    print("\n--- Three-Branch Neural Network Results ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

