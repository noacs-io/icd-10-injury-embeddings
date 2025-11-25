import logging
import os
import random
import warnings
from typing import Iterable

import numpy as np
import polars as pl
import umap
from graspologic.cluster import AutoGMMCluster
from numba.core.errors import NumbaDeprecationWarning
from pandas import DataFrame
from polars import col
from sklearn.metrics import silhouette_samples

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.cluster._agglomerative")
warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings(
    "ignore",
    message="X contains a zero vector, will not run cosine affinity.",
    category=UserWarning,
    module="graspologic.cluster.autogmm",
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s: %(message)s")

log = logging.getLogger(__name__)


MISSING_AVERAGES = {
    "T291": ["T201", "T205", "T211", "T215", "T221", "T225", "T231", "T235", "T241", "T245", "T251", "T255"],
    "T292": [
        "T201",
        "T205",
        "T211",
        "T215",
        "T221",
        "T225",
        "T231",
        "T235",
        "T241",
        "T245",
        "T251",
        "T255",
        "T202",
        "T206",
        "T212",
        "T216",
        "T222",
        "T226",
        "T232",
        "T236",
        "T242",
        "T246",
        "T252",
        "T256",
    ],
    "T293": [
        "T201",
        "T205",
        "T211",
        "T215",
        "T221",
        "T225",
        "T231",
        "T235",
        "T241",
        "T245",
        "T251",
        "T255",
        "T202",
        "T206",
        "T212",
        "T216",
        "T222",
        "T226",
        "T232",
        "T236",
        "T242",
        "T246",
        "T252",
        "T256",
        "T203",
        "T207",
        "T213",
        "T217",
        "T223",
        "T227",
        "T233",
        "T237",
        "T243",
        "T247",
        "T253",
        "T257",
    ],
    "T295": ["T205", "T215", "T225", "T235", "T245", "T255"],
    "T296": ["T205", "T215", "T225", "T235", "T245", "T255", "T206", "T216", "T226", "T236", "T246", "T256"],
    "T297": [
        "T205",
        "T215",
        "T225",
        "T235",
        "T245",
        "T255",
        "T206",
        "T216",
        "T226",
        "T236",
        "T246",
        "T256",
        "T207",
        "T217",
        "T227",
        "T237",
        "T247",
        "T257",
    ],
    "T29": ["T29"],
    "T290": ["T290", "T291", "T292", "T293"],
    "T294": ["T295", "T296", "T297"],
}


def average_patient_embedding(df: pl.DataFrame, df_codes: pl.DataFrame, embedding_dim: int):
    df_with_id = df.with_row_index()

    # Explode codes and join with embeddings
    df_joined = df_with_id.explode("codes").join(df_codes, left_on="codes", right_on="code", how="left")

    # Group by original index and calculate mean embeddings
    mean_embeddings = df_joined.group_by().agg(pl.col("embedding").mean()).drop("index")

    return mean_embeddings


def mean_embedding(df: pl.DataFrame, codes: list[str]) -> np.ndarray:
    # Filter rows where code is in the list and embedding is not null
    filtered = df.filter((pl.col("code").is_in(codes)) & (~pl.col("embedding").is_null()))

    # Extract embeddings as a list and convert to numpy array
    embeddings = filtered.select("embedding").to_numpy().flatten()

    if len(embeddings) > 0:
        return np.mean(embeddings, axis=0)
    return None


def handle_missing_codes(df_codes: pl.DataFrame) -> pl.DataFrame:
    # Find codes with missing embeddings
    missing_df = df_codes.filter(pl.col("embedding").is_null())
    missing_codes = missing_df.select("code").to_series().to_list()

    # Create dataframes to track processed results
    missing_list = []
    found_codes = []

    for code in missing_codes:
        embedding = None

        if len(code) == 5:
            # Try 4-char prefix match first
            similar_codes = (
                df_codes.filter(pl.col("code").str.contains(f"^{code[:4]}[0-9]?$")).select("code").to_series().to_list()
            )

            embedding = mean_embedding(df_codes, similar_codes)

            # If no embedding found, try 3-char prefix
            if embedding is None:
                similar_codes = (
                    df_codes.filter(pl.col("code").str.contains(f"^{code[:3]}[0-9]?$"))
                    .select("code")
                    .to_series()
                    .to_list()
                )
                embedding = mean_embedding(df_codes, similar_codes)

        elif len(code) == 4:
            similar_codes = (
                df_codes.filter(pl.col("code").str.contains(f"^{code[:3]}[0-9]?$")).select("code").to_series().to_list()
            )
            embedding = mean_embedding(df_codes, similar_codes)

        elif len(code) == 3:
            similar_codes = (
                df_codes.filter(pl.col("code").str.contains(f"^{code}")).select("code").to_series().to_list()
            )
            embedding = mean_embedding(df_codes, similar_codes)

        if embedding is not None:
            missing_list.append(pl.DataFrame({"code": [code], "embedding": [embedding]}))
            found_codes.append(code)

    # Remove found codes from original dataframe
    df_filtered = df_codes.filter(~pl.col("code").is_in(found_codes))

    # Combine original filtered dataframe with new imputed embeddings
    if missing_list:
        df_combined = pl.concat([df_filtered] + missing_list)
    else:
        df_combined = df_filtered

    # Handle special cases from MISSING_AVERAGES
    for code, codes_to_average in MISSING_AVERAGES.items():
        if code not in missing_codes:
            print(f"WARNING: CODE {code} IS NOT MISSING.")

        # Handle edge case for T29
        if len(codes_to_average) == 1:
            codes_to_average = (
                df_combined.filter(pl.col("code").str.contains(f"^{codes_to_average[0]}"))
                .select("code")
                .to_series()
                .to_list()
            )

        embedding = mean_embedding(df_combined, codes_to_average)

        # Remove the code if it exists and add the updated version
        df_combined = df_combined.filter(pl.col("code") != code)
        df_combined = pl.concat([df_combined, pl.DataFrame({"code": [code], "embedding": [embedding]})])

    return df_combined


def calculate_mean_silhouette_score(embeddings: Iterable):
    if len(embeddings[0]) > 2:
        reducer = umap.UMAP(n_components=2)
        embeddings = reducer.fit_transform(embeddings)

    auto_gmm = AutoGMMCluster()
    clustering_result = auto_gmm.fit_predict(embeddings)

    n_clusters = len(np.unique(clustering_result))

    sil_scores = silhouette_samples(embeddings, clustering_result)

    return np.mean(sil_scores), n_clusters


def seed_everything(seed: int) -> None:
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PL_SEED_WORKERS"] = "1"


def ohe_existing_codes(icd_codes: Iterable, df_ohe_lookup: pl.DataFrame) -> np.ndarray:
    df_codes = pl.DataFrame({"code": icd_codes}).with_row_index().explode("code")

    # Filter valid codes and create sparse representation
    df_codes = df_codes.filter((pl.col("code").is_not_null()) & (pl.col("code").is_in(df_ohe_lookup["code"])))

    df_codes = df_codes.join(df_ohe_lookup, on="code")

    one_hot_matrix = np.zeros((len(icd_codes), len(df_ohe_lookup)), dtype=np.int8)

    row_col_pairs = df_codes.select(["index", "ohe_index"]).to_numpy()
    one_hot_matrix[row_col_pairs[:, 0], row_col_pairs[:, 1]] = 1

    return one_hot_matrix


def ohe(icd_codes: pl.Series) -> tuple[np.ndarray, pl.DataFrame]:
    included_codes = icd_codes.explode().unique().sort()
    included_codes = included_codes.filter(included_codes != "")

    df_ohe_lookup = pl.DataFrame({"code": included_codes}).with_row_index("ohe_index")

    df_codes = pl.DataFrame({"code": icd_codes}).with_row_index().explode("code")

    # Filter valid codes and create sparse representation
    df_codes = df_codes.filter((pl.col("code").is_not_null()) & (pl.col("code").is_in(df_ohe_lookup["code"])))

    df_codes = df_codes.join(df_ohe_lookup, on="code")

    one_hot_matrix = np.zeros((len(icd_codes), len(df_ohe_lookup)), dtype=np.int8)

    row_col_pairs = df_codes.select(["index", "ohe_index"]).to_numpy()
    one_hot_matrix[row_col_pairs[:, 0], row_col_pairs[:, 1]] = 1

    return one_hot_matrix, df_ohe_lookup
