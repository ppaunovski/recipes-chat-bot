import json
import pandas as pd
import networkx as nx

def load_data_from_json(path_to_json: str) -> list[dict]:

    with open(path_to_json, "r") as file:
        data = json.load(file)

    return data

def get_ingrs_for_recipe(recipe: dict) -> tuple[str, list[str]]:
  return recipe['title'], [ingr['name'] for ingr in recipe['ingredients']]

def gen_recipe_ingredient_data_frame(data: list[dict]) -> dict[str, list[str]]:
   data_dict = {
    'ingredient': [],
    'recipe': []
    }
   
   for recipe in data:
    recipe_name, ingrs = get_ingrs_for_recipe(recipe)
    for ingr in ingrs:
        data_dict['ingredient'].append(ingr)
        data_dict['recipe'].append(recipe_name)

   return pd.DataFrame(data_dict)

def csv_file_to_pd_df(path_to_csv: str) -> pd.DataFrame:
   return pd.read_csv(path_to_csv)


def get_nodes_and_egdes_df(composed_graph: nx.DiGraph) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes_df = pd.DataFrame({
        'Node': composed_graph.nodes()
    })
    nodes_df.columns = ['Node'] + list(nodes_df.columns[1:])
    edges_df = nx.to_pandas_edgelist(composed_graph)

    list_of_recipes_from_df = set(edges_df.query('label=="has_ingr"')['source'].values)
    nodes_df['type'] = nodes_df['Node'].apply(lambda elem: 'recipe' if elem in list_of_recipes_from_df else 'ingr')
    nodes_df = nodes_df.sort_values(by='type')

    ingr_counts = nodes_df['type'].value_counts()['ingr']
    recipe_counts = nodes_df['type'].value_counts()['recipe']
    
    nodes_df['index'] = list(range(ingr_counts)) + list(range(recipe_counts))

    return (nodes_df, edges_df)
