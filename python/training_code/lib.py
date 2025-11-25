import re
from typing import Iterable

import numpy as np
import polars as pl
import torch
import torch.nn as nn

from .utils import ohe_existing_codes


class TorchAutoEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, df_ohe_lookup=None, hidden_dims=[512, 256, 128], dropout_rate=0.5):
        super(TorchAutoEncoder, self).__init__()
        self.df_ohe_lookup = df_ohe_lookup
        if self.df_ohe_lookup is None:
            self.df_ohe_lookup = pl.read_csv("results/supported_codes.csv").rename({"index": "ohe_index"})

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[2], embedding_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[2], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[0], input_dim),
        )

    def embed(self, X: Iterable):
        self.eval()

        X = ohe_existing_codes(X, self.df_ohe_lookup)

        if np.sum(X) == 0:
            return np.nan

        X = torch.Tensor(X)

        with torch.no_grad():
            embedding = self.encoder(X)

        self.train()

        return embedding.numpy()


def load_auto_encoder(embedding_dim: int):
    model = TorchAutoEncoder(input_dim=806, embedding_dim=embedding_dim)
    state_dict = torch.load(f"results/weights/{embedding_dim}.pt")
    model.load_state_dict(state_dict)

    encoder = model.encoder

    encoder.eval()

    return encoder


def one_hot_encode_icd_codes(icd_codes, df_supported_icd_trauma_codes=None, ignore_warnings: bool = True):
    if df_supported_icd_trauma_codes is None:
        df_supported_icd_trauma_codes = pl.read_csv("results/supported_codes.csv").rename({"index": "ohe_index"})

    df_codes = (
        pl.DataFrame({"code": icd_codes}).with_row_index().with_columns(pl.col("code").str.split(" ")).explode("code")
    )

    if not ignore_warnings:
        missing_codes = df_codes.filter(
            (pl.col("code").is_not_null()) & (~pl.col("code").is_in(df_supported_icd_trauma_codes["code"]))
        )["code"].unique()

        print(
            f"Could not embed ICD code(s): {', '.join(missing_codes.to_list())}.\n\n"
            f"Refer to the supported_icd_trauma_codes to see embeddable ICD codes."
        )

    # Filter valid codes and create sparse representation
    df_codes = df_codes.filter(
        (pl.col("code").is_not_null()) & (pl.col("code").is_in(df_supported_icd_trauma_codes["code"]))
    )

    df_codes = df_codes.join(df_supported_icd_trauma_codes, on="code")

    one_hot_matrix = np.zeros((len(icd_codes), len(df_supported_icd_trauma_codes)), dtype=np.int8)

    row_col_pairs = df_codes.select(["index", "ohe_index"]).to_numpy()
    one_hot_matrix[row_col_pairs[:, 0].astype(int), row_col_pairs[:, 1].astype(int)] = 1

    return one_hot_matrix


def get_patient_embedding_autoencoder(icd_codes: list[str], dim: int = 4, ignore_warnings: bool = True):
    model = load_auto_encoder(dim)

    one_hot_encoded_matrix = one_hot_encode_icd_codes(icd_codes, None, ignore_warnings)

    zero_row_indices = np.where(np.all(one_hot_encoded_matrix == 0, axis=1))[0]

    input_tensor = torch.tensor(one_hot_encoded_matrix, dtype=torch.float32)

    embeddings = np.zeros((len(icd_codes), dim))

    with torch.no_grad():
        embeddings = model(input_tensor).numpy()

    embeddings_missing = get_patient_embedding_mean(icd_codes[zero_row_indices], dim)

    embeddings[zero_row_indices] = embeddings_missing

    return embeddings


def parse_embedding(embedding_str):
    numbers = re.findall(r"-?\d+\.?\d*", embedding_str)

    return pl.Series([float(num) for num in numbers])


def get_patient_embedding_mean(icd_codes: list[str], dim: int = 4):
    df_icd_codes_embeddings = pl.read_parquet(f"results/embeddings/{dim}.parquet")

    data = {f"embedding_{i}": df_icd_codes_embeddings["embedding"].arr.get(i) for i in range(dim)}
    data["code"] = df_icd_codes_embeddings["icd_code"]

    df_icd_codes_embeddings = pl.DataFrame(data)

    df_codes = (
        pl.DataFrame({"code": icd_codes}).with_row_index().with_columns(pl.col("code").str.split(" ")).explode("code")
    )

    df_codes = df_codes.join(df_icd_codes_embeddings, on="code").drop("code").group_by("index").mean().drop("index")

    return df_codes.to_numpy()
