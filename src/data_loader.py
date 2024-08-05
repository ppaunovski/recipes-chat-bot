import json
import pandas as pd

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
