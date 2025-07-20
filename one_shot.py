import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Specify the model and load it in half-precision
model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16 # Use half-precision
).to("cuda") # Move the model to the GPU

# --- Your One-Shot ER Prompt ---
entity1 = "iphone 15 pro 256gb blue"
entity2 = "apple iphone 15 pro smartphone 256gb"

# Create a prompt that gives the model one example
prompt = f"""
Does the following pair of products refer to the same real-world entity?
Answer with "Yes" or "No".

Product 1: amd ryzen 9 5900x cpu
Product 2: amd ryzen 9 5900x 12-core processor
Answer: Yes

Product 1: {entity1}
Product 2: {entity2}
Answer:"""

# --- Get the Model's Prediction ---
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
outputs = model.generate(input_ids)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(result) # Expected output: "Yes"
