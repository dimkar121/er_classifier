import pandas as pd
import os


def create_gold_standard_from_training_set():
    """
    Reads a WDC training file in CSV format, filters it to keep only the matching pairs
    (where label is 1), and saves them to a new 'gold_standard.csv' file.

    Instructions:
    1. Make sure this script is in the same directory as your dataset files.
    2. Ensure the 'train.csv' file exists in that directory.
    3. Run the script.
    """
    # Path to the training file containing the labeled pairs
    training_file = 'train.csv'
    output_file = 'gold_standard_matches.csv'

    # Check if the training file exists before proceeding
    if not os.path.exists(training_file):
        print(f"Error: File not found at '{training_file}'")
        print("Please place this script in the same directory as your dataset files.")
        return

    print(f"Reading labeled pairs from '{training_file}'...")

    try:
        # Read the CSV file.
        df = pd.read_csv(training_file)

        print(f"Found {len(df)} total pairs.")

        # Filter the DataFrame to keep only the rows where 'label' is 1 (which indicates a match)
        matches_df = df[df['label'] == 1].copy()

        print(f"Found {len(matches_df)} matching pairs (the gold standard).")

        # Select and rename the correct columns based on your file's structure
        # Using 'ltable_id' and 'rtable_id' as provided.
        gold_standard = matches_df[['ltable_id', 'rtable_id']]
        gold_standard.rename(columns={
            'ltable_id': 'id_source_A',
            'rtable_id': 'id_source_B'
        }, inplace=True)

        # Save the final gold standard to a new CSV file
        gold_standard.to_csv(output_file, index=False)

        print(f"Successfully created '{output_file}' with all the matching pairs.")

    except KeyError as e:
        print(f"An error occurred: A required column was not found. {e}")
        print("Please ensure your CSV file contains the columns 'label', 'ltable_id', and 'rtable_id'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    create_gold_standard_from_training_set()
