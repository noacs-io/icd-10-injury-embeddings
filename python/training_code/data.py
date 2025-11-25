import logging
import os
from typing import List

import numpy as np
import polars as pl
import pytorch_lightning as L
import torch
from polars import col
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)

"""
All create_*_dataset should return a Polars DataFrame with the columns: pid, died, codes, description.
"""


class ICDDataModule(L.LightningDataModule):
    def __init__(self, X_train, X_val, batch_size: int):
        super(ICDDataModule, self).__init__()
        self.X_train = X_train
        self.X_val = X_val
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dataset_train = ICDDataset(self.X_train)
        self.dataset_val = ICDDataset(self.X_val)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count())

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count())


class ICDDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        row = self.X[idx]
        return torch.tensor(row, dtype=torch.float32)


def create_ntdb_dataset(
    dataset_path: str = "./local/data/ntdb.anon.csv",
    icd_descriptions_path: str = "./local/data/icd_descriptions.csv",
    nrows: int = None,
    cache=True,
) -> pl.DataFrame:
    df_dataset = pl.read_csv(dataset_path, null_values=["NA"])
    df_icd_descriptions = pl.read_csv(icd_descriptions_path)

    df_dataset = (
        df_dataset.with_columns(col("icd_codes").str.split(" "))
        .explode("icd_codes")
        .join(df_icd_descriptions, left_on="icd_codes", right_on="code", how="left")
        .group_by("inc_key")
        .agg(
            # Keep all other columns (first value for each inc_key)
            *[pl.col(col).first() for col in df_dataset.columns if col not in ["inc_key", "icd_codes", "description"]],
            # Join icd_codes with spaces
            pl.col("icd_codes").str.join(" ").alias("icd_codes"),
            # Join descriptions with period + space
            pl.col("description").str.join(". ").alias("description"),
        )
        .with_columns(col("icd_codes").str.split(" ").alias("codes"))
        .drop(["icd_codes"])
        .with_columns(col("codes").list.len().alias("n_codes"))
    )

    nrows_max = len(df_dataset)
    if nrows is None:
        return df_dataset
    else:
        nrows = min(nrows, nrows_max)

        return df_dataset.sample(nrows)


df = create_ntdb_dataset()
