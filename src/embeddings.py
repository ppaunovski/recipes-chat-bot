import torch
from transformers import AutoModel
from dotenv import load_dotenv
import os
from tqdm import tqdm
import pandas as pd
from utils import get_edge_index_from_label_type
import torch_geometric

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

def get_hetero_data(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> torch_geometric.data.HeteroData:
  data = HeteroData()
  ingrs_emb = list(map(get_embedding_from_model, ingrs))

  data['ingr'].x = Tensor(np.array(ingrs_emb)).to(dtype=torch.float32)
  recipes_df = nodes_df.query('type=="recipe"')['Node'].values.tolist()

  data['recipe'].x = Tensor(np.array(list(map(get_embedding_from_model, recipes_df)))).to(dtype=torch.float32)

  node_to_id_dict = {
    k: v for k,v in nodes_df[['Node', 'index']].values
  }

  edges_df['index'] = edges_df['source'].apply(lambda node: node_to_id_dict[node])
  edges_df['index_target'] = edges_df['target'].apply(lambda node: node_to_id_dict[node])

  merged_df = edges_df

  data['recipe', 'has_ingr', 'ingr'].edge_index = Tensor(get_edge_index_from_label_type('has_ingr')).to(dtype=torch.int64)
  data['ingr', 'also_known_as', 'ingr'].edge_index =  Tensor(get_edge_index_from_label_type('also_known_as')).to(dtype=torch.int64)
  data['ingr', 'has_sub', 'ingr'].edge_index =  Tensor(get_edge_index_from_label_type('has_sub')).to(dtype=torch.int64)
  
  data['recipe']['num_nodes'] = data['recipe'].x.shape[0]
  data['ingr']['num_nodes'] = data['ingr'].x.shape[0]

  return data

