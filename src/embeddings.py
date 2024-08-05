import torch
from transformers import AutoModel
from dotenv import load_dotenv
import os
from tqdm import tqdm
import pandas as pd

load_dotenv()

model_name = os.environ['model']
trust_remote_code = os.environ['trust_remote_code']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

emb_model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code) # trust_remote_code is needed to use the encode method
emb_model = emb_model.to(device)

def get_embedding_from_model(string: str) -> list[float]:
  with torch.inference_mode():
    embeddings = emb_model.encode([string])
  return embeddings[0]

def get_corr_matrix_from_model(list_of_strings: list[str]) -> list[list[float]]:
  with torch.inference_mode():
    embeddings = emb_model.encode(list_of_strings)
  return embeddings

def get_subs_embeddings(subs_df: pd.DataFrame) -> list[list[float]]:
  subs_embeddings = []
  for index, row in tqdm(list(subs_df.iterrows())):
    all_subs = [row['Name']] + row['Substitutions'].split(',')
    try:
      all_subs.remove('nan')
    except:
      ...
    subs_embeddings.append(list(map(get_embedding_from_model, all_subs)))
  return subs_embeddings