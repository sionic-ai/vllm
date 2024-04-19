from transformers import AutoTokenizer
import re
import json

def is_korean(char):
    return re.search(r'[가-힣]', char) is not None

def is_japanese(char):
    return re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', char) is not None

# Load the tokenizer using AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("../", use_fast=False)

# Extract Korean and Japanese tokens
korean_token_ids = []
japanese_token_ids = []
vocab = tokenizer.get_vocab()
for token, index in vocab.items():
    # Decode the token
    decoded_token = tokenizer.decode(index)
    
    # Check if the token is a Korean token
    if any(is_korean(char) for char in decoded_token):
        korean_token_ids.append(index)
    
    # Check if the token is a Japanese token
    if any(is_japanese(char) for char in decoded_token):
        japanese_token_ids.append(index)

# Save the results to files
with open('korean_token_ids.json', 'w') as f:
    json.dump(korean_token_ids, f)

with open('japanese_token_ids.json', 'w') as f:
    json.dump(japanese_token_ids, f)