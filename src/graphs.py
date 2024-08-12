import pandas as pd
import networkx as nx
from sklearn.cluster import DBSCAN
from collections import defaultdict
from embeddings import get_corr_matrix_from_model, get_embedding_from_model
from sklearn.metrics.pairwise import cosine_similarity
from data_loader import gen_recipe_ingredient_data_frame
import numpy as np
from tqdm import tqdm


def generate_graph(
    data: list[dict],
    from_label: str = "recipe",
    to_label: str = "ingredient",
    label: str = "has_ingr",
) -> nx.DiGraph:
    food_recipe_graph = nx.DiGraph()
    df = gen_recipe_ingredient_data_frame(data)

    for index, row in df.iterrows():
        food_recipe_graph.add_edge(row[from_label], row[to_label], label=label)

    return food_recipe_graph


def gen_clusters_of_nodes(
    df: pd.DataFrame,
    eps: float = 0.04,
    min_samples: int = 2,
    metric: str = "precomputed",
) -> dict[int, list]:
    unique_ingrs = df["ingredient"].unique()
    corr_matrix = get_corr_matrix_from_model(unique_ingrs)
    cos_matrix = cosine_similarity(corr_matrix)
    distance_matrix = 1 - cos_matrix
    distance_matrix[distance_matrix < 0] = 0

    dbscan = DBSCAN(
        eps=eps, min_samples=min_samples, metric=metric
    )  # eps = 1 - 0.96, where 0.96 is the treshold
    labels = dbscan.fit_predict(distance_matrix)

    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append(unique_ingrs[idx])

    for cluster_id, items in clusters.items():
        if cluster_id != -1:
            print(f"Cluster {cluster_id}: {set(items)}")

    return clusters


def switch_edge(
    edges: list[tuple[str, str, dict]], node: str, index: int = 0
) -> list[tuple[str, str, str]]:
    changed_edges = []
    for edge in edges:
        x, y, data_dict = edge
        label = data_dict["label"]
        changed_edges.append((node, y, label) if index == 0 else (x, node, label))
    return changed_edges


def remove_similar_nodes(
    food_recipe_graph: nx.DiGraph, clusters: dict[int, list]
) -> nx.DiGraph:
    to_switch_matrix = [list(set(items)) for cluster_id, items in clusters.items()]

    for class_of_similar_nodes in to_switch_matrix:

        similar_nodes = class_of_similar_nodes[1:]
        main_node = class_of_similar_nodes[0]

    for node in similar_nodes:
        outgoing_edges = list(food_recipe_graph.out_edges(node, data=True))
        incoming_edges = list(food_recipe_graph.in_edges(node, data=True))

        food_recipe_graph.remove_node(node)

        changed_incoming = switch_edge(incoming_edges, main_node, 1)
        changed_outgoing = switch_edge(outgoing_edges, main_node, 0)

        for u, v, label in changed_incoming + changed_outgoing:
            food_recipe_graph.add_edge(u, v, label=label)

    return food_recipe_graph
    # ova dolgo trae


def generate_subs_graph(subs_df: pd.DataFrame) -> nx.DiGraph:
    food_subs_graph = nx.DiGraph()

    for index, row in list(subs_df.iterrows()):
        if row["Substitutions"] != "nan":
            subs = row["Substitutions"].split(",")
            for sub in subs:
                food_subs_graph.add_edge(row["Name"], sub, label="has_sub")
        if row["Also known as"] != "nan":
            akas = row["Also known as"].split(", ")
            for aka in akas:
                food_subs_graph.add_edge(row["Name"], aka, label="also_known_as")

        return food_subs_graph


def get_similar_substitution_nodes(
    subs_embeddings: list[list[float]],
    recipe_ingredients_df: pd.DataFrame,
    sub_names: np.ndarray,
    treshold: float = 0.96,
) -> list[tuple[str, str]]:
    # Step 1: Collect all ingredient embeddings and substitute embeddings
    ingr_embeddings = []
    for _, row in tqdm(recipe_ingredients_df.iterrows(), total=len(recipe_ingredients_df)):
        ingr_embeddings.append(get_embedding_from_model(row['ingredient']))

    ingr_embeddings = np.array(ingr_embeddings)

    subs_embeddings_flat = np.vstack(
        subs_embeddings
    )  # Flatten list of lists to a single matrix

    # Step 2: Normalize the embeddings
    ingr_embeddings_norm = ingr_embeddings / np.linalg.norm(
        ingr_embeddings, axis=1, keepdims=True
    )
    subs_embeddings_norm = subs_embeddings_flat / np.linalg.norm(
        subs_embeddings_flat, axis=1, keepdims=True
    )

    # Step 3: Matrix multiplication to compute cosine similarities
    cos_sim_matrix = np.dot(ingr_embeddings_norm, subs_embeddings_norm.T)

    # Step 4: Precompute the index ranges for each substitute type
    sub_indices = np.cumsum([len(emb_list) for emb_list in subs_embeddings])

    # Step 5: Filter results based on threshold
    threshold = treshold
    new_list_of_subs = []

    for i, row in tqdm(
        enumerate(recipe_ingredients_df.itertuples()), total=len(recipe_ingredients_df)
    ):
        similar_indices = np.where(cos_sim_matrix[i] > threshold)[0]
        for idx in similar_indices:
            # Find the corresponding substitute type using precomputed indices
            sub_type_idx = np.searchsorted(sub_indices, idx, side="right")
            new_list_of_subs.append((row.ingredient, sub_names[sub_type_idx]))

    return new_list_of_subs


def gen_ingredient_substitution_edges(list_of_subs: list[list[float]]) -> pd.DataFrame:
    new_edges_dict = {"Ingredient": [], "Substitution": []}

    for ingr, sub in list_of_subs:
        new_edges_dict["Ingredient"].append(ingr)
        new_edges_dict["Substitution"].append(sub)

    new_edges_df = pd.DataFrame(new_edges_dict)
    return new_edges_df


def combine_graphs(
    food_recipe_graph: nx.DiGraph,
    food_subs_graph: nx.DiGraph,
    edges_df: pd.DataFrame,
    clusters: dict[int, list],
) -> nx.DiGraph:
    composed_graph = nx.compose(food_recipe_graph, food_subs_graph)

    for index, row in edges_df.iterrows():
        ingr = row["Ingredient"]
        sub = row["Substitution"]

        for cluster_id, items in clusters.items():
            if cluster_id != -1:
                items_set = set(items)
                if ingr in items_set:
                    ingr = items[0]
                    break
                
        composed_graph.add_edge(ingr, sub, label="has_sub")

    return composed_graph
