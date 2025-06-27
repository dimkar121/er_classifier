import pandas as pd
import numpy as np
import faiss
import time
from scipy.spatial.distance import jaccard
import re
import jellyfish
import torch
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re


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




# --- 2. Define the Prompt Creation Function ---
# This function formats your product data into a prompt that Phi-3 understands well.
def create_phi3_prompt(title1, description1, product2, description2):
    """
    Creates a structured prompt for the Phi-3 model, instructing it to
    perform entity resolution and return a JSON object.

    Args:
        product_a (pd.Series): A row from a DataFrame for the first product.
        product_b (pd.Series): A row from a DataFrame for the second product.

    Returns:
        str: A fully formatted prompt ready for the model.
    """
    # The <|user|> and <|assistant|> tokens are special markers for Phi-3's chat template.
    prompt = f"""<|user|>
You are an expert entity resolution system. Analyze the two product descriptions provided below. Determine if they refer to the exact same real-world item. Your response must be a valid JSON object only, with no other text or explanation.

The JSON object must have the following structure:
{{
  "is_match": boolean,
  "confidence": "High" | "Medium" | "Low",
  "reasoning": "A brief explanation of your decision, focusing on key attributes like model number and brand."
}}

**Product A:**
- title: {title1}
- description: {description1}

**Product B:**
- title: {title2}
- description: {description2}
<|end|>
<|assistant|>
"""
    return prompt


def parse_llm_response(response_text):
    """
    Extracts and parses a JSON object from the LLM's raw output string.
    This handles cases where the model might add extra text or markdown.
    """
    # Use regex to find the JSON block, even if it's wrapped in text or markdown
    match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if not match:
        print("Warning: Could not find a JSON object in the response.")
        return None

    json_string = match.group(0)
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        print("Warning: Failed to decode the extracted JSON string.")
        return None


if __name__ == '__main__':
    print("--- 1. Loading microsoft/Phi-3-mini-4k-instruct model ---")

    model_id = "microsoft/Phi-3-mini-4k-instruct"
    try:
        # Set the device to GPU if available, otherwise CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
            #torch_dtype="auto",
            torch_dtype=torch.float32,  # Using a more standard dtype instead of "auto"
            trust_remote_code=True,
            attn_implementation="eager"
        )
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        print("Model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print(
            "Please ensure you have an internet connection and the 'transformers' and 'torch' libraries are installed.")
        exit()


    truth = pd.read_csv("./data/truth_abt_buy.csv", sep=",", encoding="unicode_escape", keep_default_na=False)
    truthD = dict()
    a = 0
    for i, r in truth.iterrows():
        idAbt = r["idAbt"]
        idBuy = r["idBuy"]
        if idAbt in truthD:
            ids = truthD[idAbt]
            ids.append(idBuy)
            a += 1
        else:
            truthD[idAbt] = [idBuy]
    matches = len(truthD.keys()) + a
    print("No of matches=", matches)

    # ====================================================================--

    from tensorflow import keras  # Or `import keras` depending on your setup

    #loaded_model_path = './data/er_abt_buy.keras'
    # Load the model
    #loaded_model = keras.models.load_model(loaded_model_path)

    #loaded_model = xgb.XGBClassifier()
    #loaded_model.load_model("./data/abt_buy_xgb_model.json")
    #print("Model loaded successfully!")
    # You can verify the model architecture
    #loaded_model.summary()
    batch_size = 10_000
    #model = "mini"
    num_candidates = 2
    d = 384
    phi = 0.1520531820505105065205350
    df11 = pd.read_parquet(f"./data/Abt_embedded_mini_ft.pqt")
    df22 = pd.read_parquet(f"./data/Buy_embedded_mini_ft.pqt")
    #df11 = pd.read_parquet(f"./data/Abt_embedded_mini.pqt")
    #df22 = pd.read_parquet(f"./data/Buy_embedded_mini.pqt")
    vectors_buy = df22['v'].tolist()
    buy_embeddings = np.array(vectors_buy).astype(np.float32)
    vectors_abt = df11['v'].tolist()
    abt_embeddings = np.array(vectors_abt).astype(np.float32)
    abt_ids = np.array(df11['id'].tolist())
    buy_ids = np.array(df22['id'].tolist())

    df1_minhash = pd.read_parquet(f"./data/Abt_embedded_minhash_all.pqt")
    df2_minhash = pd.read_parquet(f"./data/Buy_embedded_minhash_all.pqt")
    vectors_abt_minhash = np.array(df1_minhash['namev'].tolist())
    vectors_buy_minhash = np.array(df2_minhash['namev'].tolist())

    #hybrid_vectors1 = np.concatenate((abt_embeddings, vectors_abt_minhash ), axis=1)
    #hybrid_vectors2 = np.concatenate((buy_embeddings, vectors_buy_minhash), axis=1)

    #hybrid_vectors1 = hybrid_vectors1.astype('float32')
    #hybrid_vectors2 = hybrid_vectors2.astype('float32')

    #print("Shape of the final concatenated array:", hybrid_vectors1.shape)


    minhash_names1 = {row['id']: row['namev'] for index, row in df1_minhash.iterrows()}
    minhash_names2 = {row['id']: row['namev'] for index, row in df2_minhash.iterrows()}
    minhash_descrs1 = {row['id']: row['descriptionv'] for index, row in df1_minhash.iterrows()}
    minhash_descrs2 = {row['id']: row['descriptionv'] for index, row in df2_minhash.iterrows()}
    df1_minhash['models'] = df1_minhash['name'].apply(extract_model)
    df2_minhash['models'] = df2_minhash['name'].apply(extract_model)
    models1 = {row['id']: row['models'] for index, row in df1_minhash.iterrows()}
    models2 = {row['id']: row['models'] for index, row in df2_minhash.iterrows()}
    names1 = {row['id']: row['name'] for index, row in df1_minhash.iterrows()}
    names2 = {row['id']: row['name'] for index, row in df2_minhash.iterrows()}
    prices1 = {row['id']: row['price'] for index, row in df1_minhash.iterrows()}
    prices2 = {row['id']: row['price'] for index, row in df2_minhash.iterrows()}
    all_brands = df2_minhash['brand'].dropna().unique()
    brands_list = sorted([str(b).lower() for b in all_brands if len(str(b)) > 2], key=len, reverse=True)
    df1_minhash['brand'] = df1_minhash['name'].apply(lambda text: find_brand_in_text(text, brands_list))
    brands1 = {row['id']: row['brand'] for index, row in df1_minhash.iterrows()}
    brands2 = {row['id']: row['brand'] for index, row in df2_minhash.iterrows()}

    d = abt_embeddings.shape[1]
    index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_L2)
    index.hnsw.efConstruction = 60
    index.hnsw.efSearch = 16
    index.add(abt_embeddings)

    tp = 0
    fp = 0
    start_time = time.time()

    df_abt_indexed = df11.set_index('id')
    df_buy_indexed = df22.set_index('id')
    for i in range(0, len(vectors_buy), batch_size):

        buys = np.array(vectors_buy[i: i + batch_size]).astype(np.float32)
        buy_ids_in_batch = buy_ids[i: i + batch_size]
        distances, candidate_indices = index.search(buys, num_candidates)  # return scholars
        flat_candidate_ids = candidate_indices.flatten()
        candidate_abt_embeddings = abt_embeddings[flat_candidate_ids]
        repeated_buy_embeddings = np.repeat(buys, num_candidates, axis=0)
        repeated_buy_ids = np.repeat(buy_ids_in_batch, num_candidates)
        abt_ids_in_batch = abt_ids[flat_candidate_ids]
        for abt_ind, buyId in zip(candidate_indices.flatten(), repeated_buy_ids):
          abtId = abt_ids[abt_ind]
          title1 = df_abt_indexed.loc[abtId, 'name']
          description1 = df_abt_indexed.loc[abtId, 'description']
          title2 = df_buy_indexed.loc[buyId, 'name']
          description2 = df_buy_indexed.loc[buyId, 'description']

          # Create the prompt
          prompt = create_phi3_prompt(title1, description1, title2, description2)

          print("\n--- 4. Sending prompt to the model ---")

          inputs = tokenizer(prompt, return_tensors="pt").to(device)
          outputs = model.generate(**inputs, max_new_tokens=100, use_cache=False)


          # Decode the output tokens back to text
          # The [0] accesses the first (and only) sequence in the batch
          generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
          print("\n--- 5. Received raw response from model ---")
          print(generated_text)

          # Extract just the assistant's part of the response
          assistant_response = generated_text.split("<|assistant|>")[-1]

          # Parse the response to get the structured JSON
          result = parse_llm_response(assistant_response)

          print("\n--- 6. Final Parsed Result ---")
          if result:
              # Use json.dumps for pretty printing the dictionary
              print(json.dumps(result, indent=2))
          else:
              print("Could not get a valid result from the model.")

          if result["is_match"]:
            tpFound = False
            if abtId in truthD.keys():
                idBuys = truthD[abtId]
                for idBuy in idBuys:
                   if idBuy == buyId:
                       tp += 1
                       tpFound=True
                if not tpFound:
                      fp += 1


    end_time = time.time()
    print(f"recall={round(tp / matches, 2)} precision={round(tp / (tp + fp), 2)} total matching time={end_time-start_time} seconds.")
