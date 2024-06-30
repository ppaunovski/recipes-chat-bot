import pandas as pd
import re

list_of_types = [
    "flavorings",
    "accompaniments",
    "baked-goods",
    "baking-supplies",
    "dairy",
    "equipment",
    "vegetarian",
    "vegetables",
    "miscellaneous",
    "fats-oils",
    "fish",
    "fruit",
    "grain-products",
    "grains",
    "legumes-nuts",
    "liquids",
    "meats",
]


def parse_list_to_str(entry: str) -> str:
    if len(entry) <= 2:
        return ""

    pattern = r"'(.*?)'"
    matches = re.findall(pattern, entry)

    return ",".join(match for match in matches)


def parse_data_frame_substitution_column(data_frame_name: str) -> pd.DataFrame:

    df = pd.read_csv(f"data/scrapped_{data_frame_name}.csv")
    df["Substitutions"] = df["Substitutions"].apply(parse_list_to_str)

    df = df.drop(labels=["Unnamed: 0"], axis=1)
    df.to_csv(f"data/wrangled/{data_frame_name}_wrangled.csv", index=False)

    return df


def merge_all_wrangeld_data_frames(list_of_subs: list[str]) -> pd.DataFrame:

    list_of_dfs = [
        pd.read_csv(f"data/wrangled/{data_frame_name}_wrangled.csv")
        for data_frame_name in list_of_subs
    ]

    list_of_food_types = []
    for df, sub_type in zip(list_of_dfs, list_of_subs):
        list_of_food_types.extend([sub_type for _ in range(len(df))])

    df = pd.concat(list_of_dfs, axis=0)
    df["Food Type"] = list_of_food_types

    df.to_csv(f"data/wrangled/whole_wrangled.csv", index=False)

    return df


if __name__ == "__main__":

    df = merge_all_wrangeld_data_frames(list_of_subs=list_of_types)

    print(df.sample(10))
