import requests
import os
import pandas as pd
from transformers import  AutoTokenizer
from datasets import Dataset
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
url = "https://huggingface.co/datasets/bigcode/starcoderdata/resolve/main/python/train-00000-of-00059.parquet"
output_path = os.path.join(os.path.dirname(__file__), "data/starcoder/train-00000-of-00059.parquet")

headers = {"Authorization": f"Bearer {hf_token}"}
response = requests.get(url, headers=headers, stream=True)

if response.status_code == 200:
  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  with open(output_path, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
      f.write(chunk)
  print(f"File successfully downloaded: {output_path}")


model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True,  map_device="auto", add_eos_token=True, use_fast=True)


df = pd.read_parquet(output_path)
df = df.sample(10000, random_state=42, ignore_index=True)

df['input_ids'] = df['content'].apply(lambda x: tokenizer(x, return_tensors="pt").input_ids[0].tolist())
df['input_ids_size'] = df['input_ids'].apply(lambda x: len(x))


hf_dataset = Dataset.from_pandas(df)
hf_dataset.push_to_hub("starcoder", token=hf_token)