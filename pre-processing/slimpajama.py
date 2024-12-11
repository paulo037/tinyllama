import os
import pandas as pd
import zstandard as zstd
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from datasets import Dataset
from transformers import AutoTokenizer
import time


load_dotenv()


hf_token = os.getenv("HF_TOKEN")

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(
    model_id, trust_remote_code=True,  map_device="auto", add_eos_token=True, use_fast=True)


def download_folder(model_id, folder_path, repo_type="model", output_dir="data"):

    for i in range(50):
      file_name = f"/example_train_{i}.jsonl.zst"
      hf_hub_download(repo_id=model_id, filename=(f'{folder_path}{file_name}'), local_dir=output_dir, repo_type=repo_type)
      if (i + 1) % 10 == 0:
        time.sleep(60)

output_dir = os.path.join(os.path.dirname(__file__), "data/slimpajama")
chunk = "train/chunk1"
download_folder("cerebras/SlimPajama-627B", chunk, "dataset",  output_dir=output_dir)




def read_zst_jsonl(file_path):
    with open(file_path, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        
        decompressed_data = dctx.stream_reader(f)
        return pd.read_json(decompressed_data, lines=True)


folder_path = os.path.join(output_dir, chunk)
files = [f for f in os.listdir(folder_path) if f.endswith('.jsonl.zst')]
df = pd.concat([read_zst_jsonl(os.path.join(folder_path, file)) for file in files], ignore_index=True)


df = df.sample(50000, random_state=42, ignore_index=True)


df['input_ids'] = df['text'].apply(lambda x: tokenizer(x, return_tensors="pt").input_ids[0].tolist())


hf_dataset = Dataset.from_pandas(df)
print(hf_dataset)
hf_dataset.push_to_hub("slimpajama", token=hf_token)