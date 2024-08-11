import numpy as np
import pandas as pd
from numpy.linalg import norm

cos_sim = lambda a, b: (a @ b.T) / (norm(a) * norm(b))


def get_edge_index_from_label_type(merged_df: pd.DataFrame, label: str) -> np.array:
    return (
        merged_df[["index", "label", "index_target"]]
        .query("label==" + '"' + label + '"')[["index", "index_target"]]
        .values.reshape(2, -1)
    )
