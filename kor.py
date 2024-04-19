import json
from transformers import AutoTokenizer
import re

def is_korean_token(token):
    print("decoded_token", token)
    # Check if the token contains Korean characters
    return re.search(r'[가-힣]', token) is not None

# Load the tokenizer using AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("../", use_fast=False)

# Extract Korean tokens
korean_tokens = {}
vocab = tokenizer.get_vocab()
for token, index in vocab.items():
    # Decode the token
    decoded_token = tokenizer.decode(index)
    
    # Check if the token is a Korean token
    if is_korean_token(decoded_token):
        korean_tokens[decoded_token] = index

# Print or save the results
print(korean_tokens)
# Or
with open('korean_tokens.json', 'w', encoding='utf-8') as f:
    json.dump(korean_tokens, f, ensure_ascii=False, indent=4)