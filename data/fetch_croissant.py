import os
import json
import requests
from dotenv import load_dotenv

load_dotenv("../access_tokens.env")

headers = {"Authorization": f"Bearer {os.getenv("HF_TOKEN")}"}
API_URL = "https://huggingface.co/api/datasets/aplesner-eth/FLIP-Challenge/croissant"
def query():
    response = requests.get(API_URL, headers=headers)
    return response.json()
data = query()

# Export the data to a JSON file
with open("croissant.json", "w") as f:
    json.dump(data, f, indent=4)
