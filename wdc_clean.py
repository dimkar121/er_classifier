import pandas as pd
import numpy as np
import faiss
import re
from tqdm import tqdm


def find_near_duplicates_to_discard(df_full, gold_standard_ids, distance_threshold=0.1, k_neighbors=5):
    """
    Finds IDs of records that are near-duplicates to a given set of 'gold' records.

    Args:
        df_full (pd.DataFrame): The full dataframe to search within (e.g., all of Table A).
        gold_standard_ids (set): A set of IDs that are part of known matches.
        distance_threshold (float): The L2 distance below which a neighbor is considered a near-duplicate.
        k_neighbors (int): The number of neighbors to check for each gold standard record.

    Returns:
        set: A set of IDs to be discarded.
    """
    print(f"Finding near-duplicates within a table of {len(df_full)} records...")

    # Build a FAISS index on the entire table
    embeddings = np.array(df_full['v'].tolist()).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    ids_to_discard = set()

    # Create a mapping from ID to row index for quick lookup
    id_to_idx = {id_val: i for i, id_val in enumerate(df_full['id'])}

    # For each entity that is part of the gold standard...
    for gold_id in tqdm(gold_standard_ids, desc="Checking for near-duplicates"):
        anchor_idx = id_to_idx.get(gold_id)
        if anchor_idx is None:
            continue

        anchor_embedding = embeddings[anchor_idx].reshape(1, -1)

        # Search the index for its nearest neighbors
        distances, neighbor_indices = index.search(anchor_embedding, k_neighbors)

        # Check each neighbor
        for i, neighbor_idx in enumerate(neighbor_indices[0]):
            neighbor_id = df_full.iloc[neighbor_idx]['id']

            # Skip the entity itself
            if neighbor_id == gold_id:
                continue

            # If the neighbor is very close (below the threshold), it's a near-duplicate.
            # We also ensure we don't discard another gold standard record by mistake.
            distance = distances[0][i]
            if distance < distance_threshold and neighbor_id not in gold_standard_ids:
                ids_to_discard.add(neighbor_id)

    return ids_to_discard


def clean_near_duplicates_from_sources(
        file_a_csv, file_a_pqt,
        file_b_csv, file_b_pqt,
        file_gold_standard,
        distance_threshold=0.1
):
    """
    Cleans source tables by removing records that are near-duplicates of
    gold standard entities, then overwrites the original files.
    """
    print("--- Starting Near-Duplicate Cleaning Process ---")

    # --- Load all source files ---
    try:
        print("Loading source files...")
        df_a_csv = pd.read_csv(file_a_csv)
        df_a_pqt = pd.read_parquet(file_a_pqt)
        df_b_csv = pd.read_csv(file_b_csv)
        df_b_pqt = pd.read_parquet(file_b_pqt)
        df_gold = pd.read_csv(file_gold_standard)
        print("All files loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. {e}")
        return

    # --- Ensure all ID columns are strings for consistent merging and filtering ---
    for df in [df_a_csv, df_a_pqt, df_b_csv, df_b_pqt]:
        if 'id' in df.columns:
            df['id'] = df['id'].astype(str)

    for col in ['id_A', 'id_B']:
        if col in df_gold.columns:
            df_gold[col] = df_gold[col].astype(str)

    # --- Create TEMPORARY merged DataFrames for the cleaning logic ---
    print("Creating temporary merged dataframes for analysis...")
    df_a_full_temp = pd.merge(df_a_csv, df_a_pqt, on='id', how='inner')
    df_b_full_temp = pd.merge(df_b_csv, df_b_pqt, on='id', how='inner')

    # --- Identify the set of all IDs involved in the gold standard ---
    gold_standard_a_ids = set(df_gold['id_A'])
    gold_standard_b_ids = set(df_gold['id_B'])

    # --- Find near-duplicates to discard from each table ---
    ids_to_discard_a = find_near_duplicates_to_discard(df_a_full_temp, gold_standard_a_ids, distance_threshold)
    ids_to_discard_b = find_near_duplicates_to_discard(df_b_full_temp, gold_standard_b_ids, distance_threshold)

    print(f"\nFlagged {len(ids_to_discard_a)} records for removal from Table A.")
    print(f"Flagged {len(ids_to_discard_b)} records for removal from Table B.")

    # --- Filter the ORIGINAL DataFrames using the discard lists ---
    df_a_csv_clean = df_a_csv[~df_a_csv['id'].isin(ids_to_discard_a)].copy()
    df_a_pqt_clean = df_a_pqt[~df_a_pqt['id'].isin(ids_to_discard_a)].copy()

    df_b_csv_clean = df_b_csv[~df_b_csv['id'].isin(ids_to_discard_b)].copy()
    df_b_pqt_clean = df_b_pqt[~df_b_pqt['id'].isin(ids_to_discard_b)].copy()

    print(f"\nNew size of Table A CSV: {len(df_a_csv_clean)} rows.")
    print(f"New size of Table B CSV: {len(df_b_csv_clean)} rows.")

    # --- Save the cleaned files, overwriting the originals ---
    print("\nOverwriting original files with cleaned versions...")

    df_a_csv_clean.to_csv(file_a_csv, index=False)
    df_b_csv_clean.to_csv(file_b_csv, index=False)
    df_a_pqt_clean.to_parquet(file_a_pqt)
    df_b_pqt_clean.to_parquet(file_b_pqt)

    print("\n--- Process Complete! ---")


if __name__ == '__main__':
    # --- Define your input filenames here ---
    # These files will be read and then overwritten with the cleaned versions.
    INPUT_A_CSV = 'trainA.csv'
    INPUT_A_PQT = 'trainA.pqt'
    INPUT_B_CSV = 'trainB.csv'
    INPUT_B_PQT = 'trainB.pqt'
    # This file should only contain the columns 'idA' and 'idB' for true matches
    INPUT_GOLD_STANDARD = 'gold_standard.csv'

    # Call the main function with your defined file paths
    clean_near_duplicates_from_sources(
        file_a_csv=INPUT_A_CSV,
        file_a_pqt=INPUT_A_PQT,
        file_b_csv=INPUT_B_CSV,
        file_b_pqt=INPUT_B_PQT,
        file_gold_standard=INPUT_GOLD_STANDARD,
        distance_threshold=0.1  # Tune this value: smaller is stricter
    )
