import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import embed_features
import wdc_fine_tune

def preprocess_datasets(
        file_table_a='./data/wdc/tableA.csv',
        file_table_b='./data/wdc/tableB.csv',
        file_train='./data/wdc/train.csv',
        file_valid='./data/wdc/valid.csv',
        file_test='./data/wdc/test.csv'
):
    """
    Filters Table A and Table B to only include entities present in the
    train, validation, or test sets. Also creates a gold standard file
    of all matching pairs from the train, valid, and test sets.
    """
    print("--- Starting Data Preprocessing ---")

    # --- 1. Load all source files ---
    try:
        print("Loading source files...")
        df_a = pd.read_csv(file_table_a)
        df_b = pd.read_csv(file_table_b)
        df_train = pd.read_csv(file_train)
        df_valid = pd.read_csv(file_valid)
        df_test = pd.read_csv(file_test)
        print("All files loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. {e}")
        print("Please ensure tableA.csv, tableB.csv, train.csv, valid.csv, and test.csv are in the same directory.")
        return

    # --- 2. Create the Gold Standard from all splits (train, valid, test) ---
    print("\n--- Creating Gold Standard from all data splits ---")

    # Combine the train, validation, and test dataframes
    all_labeled_pairs = pd.concat([df_train, df_valid, df_test], ignore_index=True)

    # Filter for rows where the label indicates a match
    gold_standard_df = all_labeled_pairs[all_labeled_pairs['label'] == 1].copy()

    # Select and rename the ID columns for clarity
    gold_standard_df = gold_standard_df[['ltable_id', 'rtable_id']].rename(
        columns={'ltable_id': 'id_A', 'rtable_id': 'id_B'}
    )

    # Save the gold standard to a new CSV file
    gold_standard_output_path = './data/wdc/gold_standard.csv'
    gold_standard_df.to_csv(gold_standard_output_path, index=False)
    print(f"Successfully created '{gold_standard_output_path}' with {len(gold_standard_df)} matching pairs.")

    # --- 3. Collect all relevant IDs from all splits ---
    print("\n--- Collecting all relevant entity IDs ---")

    # The 'all_labeled_pairs' dataframe already contains all pairs
    relevant_a_ids = set(all_labeled_pairs['ltable_id'])
    relevant_b_ids = set(all_labeled_pairs['rtable_id'])

    print(f"Found {len(relevant_a_ids)} unique entities from Table A to keep.")
    print(f"Found {len(relevant_b_ids)} unique entities from Table B to keep.")

    # --- 4. Filter the main tables based on the collected IDs ---
    print("\n--- Filtering main tables ---")

    # Filter df_a where its 'id' is in our set of relevant IDs
    df_a_filtered = df_a[df_a['id'].isin(relevant_a_ids)].copy()

    # Filter df_b where its 'id' is in our set of relevant IDs
    df_b_filtered = df_b[df_b['id'].isin(relevant_b_ids)].copy()

    # --- 5. Save the new, filtered tables ---
    print("\n--- Saving filtered tables ---")

    table_a_output_path = './data/wdc/tableA_.csv'
    table_b_output_path = './data/wdc/tableB_.csv'

    df_a_filtered.to_csv(table_a_output_path, index=False)
    df_b_filtered.to_csv(table_b_output_path, index=False)

    all_labeled_pairs = pd.concat([df_test], ignore_index=True)

    # Filter for rows where the label indicates a match
    gold_standard_df = all_labeled_pairs[all_labeled_pairs['label'] == 1].copy()

    # Select and rename the ID columns for clarity
    gold_standard_df = gold_standard_df[['ltable_id', 'rtable_id']].rename(
        columns={'ltable_id': 'id_A', 'rtable_id': 'id_B'}
    )

    # Save the gold standard to a new CSV file
    gold_standard_output_path = './data/wdc/gold_standard.csv'
    gold_standard_df.to_csv(gold_standard_output_path, index=False)
    print(f"Successfully created '{gold_standard_output_path}' with {len(gold_standard_df)} matching pairs.")

    print(f"Successfully created '{table_a_output_path}' with {len(df_a_filtered)} rows.")
    print(f"Successfully created '{table_b_output_path}' with {len(df_b_filtered)} rows.")
    print("\n--- Preprocessing Complete! ---")


def create_test_set_tables(
        file_table_a='./data/wdc/tableA.csv',
        file_table_b='./data/wdc/tableB.csv',
        file_test='./data/wdc/test.csv'
):
    """
    Filters Table A and Table B to only include entities that appear
    in the test.csv file and saves them as Parquet files.
    """
    print("--- Starting Test Set Table Creation ---")

    # --- 1. Load all source files ---
    try:
        print("Loading source files...")
        df_a = pd.read_csv(file_table_a)
        df_b = pd.read_csv(file_table_b)
        df_test = pd.read_csv(file_test)
        print("All files loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. {e}")
        print("Please ensure tableA.csv, tableB.csv, and test.csv are in the same directory.")
        return

    # --- 2. Collect all unique IDs from the test set ---
    print("\n--- Collecting IDs from the test set ---")

    # Get all unique IDs for the left table (Table A) from test.csv
    test_a_ids = set(df_test['ltable_id'])

    # Get all unique IDs for the right table (Table B) from test.csv
    test_b_ids = set(df_test['rtable_id'])

    print(f"Found {len(test_a_ids)} unique entities from Table A in the test set.")
    print(f"Found {len(test_b_ids)} unique entities from Table B in the test set.")

    # --- 3. Filter the main tables based on the collected IDs ---
    print("\n--- Filtering main tables ---")

    # Filter df_a where its 'id' is in our set of test IDs
    df_a_filtered = df_a[df_a['id'].isin(test_a_ids)].copy()

    # Filter df_b where its 'id' is in our set of test IDs
    df_b_filtered = df_b[df_b['id'].isin(test_b_ids)].copy()

    # --- 4. Save the new, filtered tables to Parquet format ---

    all_labeled_pairs = pd.concat([df_test], ignore_index=True)

    # Filter for rows where the label indicates a match
    gold_standard_df = all_labeled_pairs[all_labeled_pairs['label'] == 1].copy()

    # Select and rename the ID columns for clarity
    gold_standard_df = gold_standard_df[['ltable_id', 'rtable_id']].rename(
        columns={'ltable_id': 'id_A', 'rtable_id': 'id_B'}
    )

    # Save the gold standard to a new CSV file
    gold_standard_output_path = './data/wdc/gold_standard_test.csv'
    gold_standard_df.to_csv(gold_standard_output_path, index=False)
    print(f"Successfully created '{gold_standard_output_path}' with {len(gold_standard_df)} matching pairs.")

    table_a_output_path = './data/wdc/tableA_test.csv'
    table_b_output_path = './data/wdc/tableB_test.csv'

    df_a_filtered.to_csv(table_a_output_path, index=False)
    df_b_filtered.to_csv(table_b_output_path, index=False)

    print(f"Successfully created '{table_a_output_path}' with {len(df_a_filtered)} rows.")
    print(f"Successfully created '{table_b_output_path}' with {len(df_b_filtered)} rows.")
    print("\n--- Process Complete! ---")


def create_train_set_tables(
        file_table_a='./data/wdc/tableA_.csv',
        file_table_b='./data/wdc/tableB_.csv',
        file_train='./data/wdc/train.csv'
):
    """
    Filters Table A and Table B to only include entities that appear
    in the test.csv file and saves them as Parquet files.
    """
    print("--- Starting Train Set Table Creation ---")

    # --- 1. Load all source files ---
    try:
        print("Loading source files...")
        df_a = pd.read_csv(file_table_a)
        df_b = pd.read_csv(file_table_b)
        df_train = pd.read_csv(file_train)
        print("All files loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. {e}")
        print("Please ensure tableA.csv, tableB.csv, and test.csv are in the same directory.")
        return

    # --- 2. Collect all unique IDs from the test set ---
    print("\n--- Collecting IDs from the train set ---")

    # Get all unique IDs for the left table (Table A) from test.csv
    train_a_ids = set(df_train['ltable_id'])

    # Get all unique IDs for the right table (Table B) from test.csv
    train_b_ids = set(df_train['rtable_id'])

    print(f"Found {len(train_a_ids)} unique entities from Table A in the train set.")
    print(f"Found {len(train_b_ids)} unique entities from Table B in the train set.")

    # --- 3. Filter the main tables based on the collected IDs ---
    print("\n--- Filtering main tables ---")

    # Filter df_a where its 'id' is in our set of test IDs
    df_a_filtered = df_a[df_a['id'].isin(train_a_ids)].copy()

    # Filter df_b where its 'id' is in our set of test IDs
    df_b_filtered = df_b[df_b['id'].isin(train_b_ids)].copy()

    # --- 4. Save the new, filtered tables to Parquet format ---

    all_labeled_pairs = pd.concat([df_train], ignore_index=True)

    # Filter for rows where the label indicates a match
    gold_standard_df = all_labeled_pairs[all_labeled_pairs['label'] == 1].copy()

    # Select and rename the ID columns for clarity
    gold_standard_df = gold_standard_df[['ltable_id', 'rtable_id']].rename(
        columns={'ltable_id': 'id_A', 'rtable_id': 'id_B'}
    )

    # Save the gold standard to a new CSV file
    gold_standard_output_path = './data/wdc/gold_standard_train.csv'
    gold_standard_df.to_csv(gold_standard_output_path, index=False)
    print(f"Successfully created '{gold_standard_output_path}' with {len(gold_standard_df)} matching pairs.")

    table_a_output_path = './data/wdc/tableA_train.csv'
    table_b_output_path = './data/wdc/tableB_train.csv'

    df_a_filtered.to_csv(table_a_output_path, index=False)
    df_b_filtered.to_csv(table_b_output_path, index=False)

    print(f"Successfully created '{table_a_output_path}' with {len(df_a_filtered)} rows.")
    print(f"Successfully created '{table_b_output_path}' with {len(df_b_filtered)} rows.")
    print("\n--- Process Complete! ---")

def create_valid_set_tables(
        file_table_a='./data/wdc/tableA_.csv',
        file_table_b='./data/wdc/tableB_.csv',
        file_train='./data/wdc/valid.csv'
):
    """
    Filters Table A and Table B to only include entities that appear
    in the test.csv file and saves them as Parquet files.
    """
    print("--- Starting Train Set Table Creation ---")

    # --- 1. Load all source files ---
    try:
        print("Loading source files...")
        df_a = pd.read_csv(file_table_a)
        df_b = pd.read_csv(file_table_b)
        df_valid = pd.read_csv(file_train)
        print("All files loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. {e}")
        print("Please ensure tableA.csv, tableB.csv, and test.csv are in the same directory.")
        return

    # --- 2. Collect all unique IDs from the test set ---
    print("\n--- Collecting IDs from the train set ---")

    # Get all unique IDs for the left table (Table A) from test.csv
    valid_a_ids = set(df_valid['ltable_id'])

    # Get all unique IDs for the right table (Table B) from test.csv
    valid_b_ids = set(df_valid['rtable_id'])

    print(f"Found {len(valid_a_ids)} unique entities from Table A in the train set.")
    print(f"Found {len(valid_b_ids)} unique entities from Table B in the train set.")

    # --- 3. Filter the main tables based on the collected IDs ---
    print("\n--- Filtering main tables ---")

    # Filter df_a where its 'id' is in our set of test IDs
    df_a_filtered = df_a[df_a['id'].isin(valid_a_ids)].copy()

    # Filter df_b where its 'id' is in our set of test IDs
    df_b_filtered = df_b[df_b['id'].isin(valid_b_ids)].copy()

    # --- 4. Save the new, filtered tables to Parquet format ---

    all_labeled_pairs = pd.concat([df_valid], ignore_index=True)

    # Filter for rows where the label indicates a match
    gold_standard_df = all_labeled_pairs[all_labeled_pairs['label'] == 1].copy()

    # Select and rename the ID columns for clarity
    gold_standard_df = gold_standard_df[['ltable_id', 'rtable_id']].rename(
        columns={'ltable_id': 'id_A', 'rtable_id': 'id_B'}
    )

    # Save the gold standard to a new CSV file
    gold_standard_output_path = './data/wdc/gold_standard_valid.csv'
    gold_standard_df.to_csv(gold_standard_output_path, index=False)
    print(f"Successfully created '{gold_standard_output_path}' with {len(gold_standard_df)} matching pairs.")

    table_a_output_path = './data/wdc/tableA_valid.csv'
    table_b_output_path = './data/wdc/tableB_valid.csv'

    df_a_filtered.to_csv(table_a_output_path, index=False)
    df_b_filtered.to_csv(table_b_output_path, index=False)

    print(f"Successfully created '{table_a_output_path}' with {len(df_a_filtered)} rows.")
    print(f"Successfully created '{table_b_output_path}' with {len(df_b_filtered)} rows.")
    print("\n--- Process Complete! ---")



if __name__ == '__main__':
    # This block runs when the script is executed directly
    preprocess_datasets()
    model_name = 'all-MiniLM-L6-v2'
    model_name = "all-mpnet-base-v2"
    embedding_model = SentenceTransformer(model_name)
    embed_features.process_and_embed_table(
        input_filename='./data/wdc/tableA_.csv',
        output_filename='./data/wdc/tableA_.pqt',
        model=embedding_model
    )

    embed_features.process_and_embed_table(
        input_filename='./data/wdc/tableB_.csv',
        output_filename='./data/wdc/tableB_.pqt',
        model=embedding_model
    )


    create_test_set_tables()
    model_name = 'all-MiniLM-L6-v2'
    model_name = "all-mpnet-base-v2"
    embedding_model = SentenceTransformer(model_name)
    embed_features.process_and_embed_table(
      input_filename='./data/wdc/tableA_test.csv',
      output_filename='./data/wdc/tableA_test.pqt',
      model=embedding_model
    )
    embed_features.process_and_embed_table(
        input_filename='./data/wdc/tableB_test.csv',
        output_filename='./data/wdc/tableB_test.pqt',
        model=embedding_model
    )

    create_train_set_tables()
    model_name = 'all-MiniLM-L6-v2'
    model_name = "all-mpnet-base-v2"
    embedding_model = SentenceTransformer(model_name)
    embed_features.process_and_embed_table(
        input_filename='./data/wdc/tableA_train.csv',
        output_filename='./data/wdc/tableA_train.pqt',
        model=embedding_model
    )
    embed_features.process_and_embed_table(
        input_filename='./data/wdc/tableB_train.csv',
        output_filename='./data/wdc/tableB_train.pqt',
        model=embedding_model
    )

    wdc_fine_tune.fine_tune()
    model_path = './data/wdc/wdc-finetuned-model'
    embedding_model = SentenceTransformer(model_path)
    embed_features.process_and_embed_table(
        input_filename='./data/wdc/tableA_train.csv',
        output_filename='./data/wdc/tableA_train_tuned.pqt',
        model=embedding_model
    )
    embed_features.process_and_embed_table(
        input_filename='./data/wdc/tableB_train.csv',
        output_filename='./data/wdc/tableB_train_tuned.pqt',
        model=embedding_model
    )

    create_test_set_tables()
    embedding_model = SentenceTransformer(model_path)
    embed_features.process_and_embed_table(
        input_filename='./data/wdc/tableA_test.csv',
        output_filename='./data/wdc/tableA_test_tuned.pqt',
        model=embedding_model
    )
    embed_features.process_and_embed_table(
        input_filename='./data/wdc/tableB_test.csv',
        output_filename='./data/wdc/tableB_test_tuned.pqt',
        model=embedding_model
    )

    create_valid_set_tables()
    embedding_model = SentenceTransformer(model_path)
    embed_features.process_and_embed_table(
        input_filename='./data/wdc/tableA_valid.csv',
        output_filename='./data/wdc/tableA_valid_tuned.pqt',
        model=embedding_model
    )
    embed_features.process_and_embed_table(
        input_filename='./data/wdc/tableB_valid.csv',
        output_filename='./data/wdc/tableB_valid_tuned.pqt',
        model=embedding_model
    )