import logging
import os
from itertools import combinations
from typing import Iterable

import gensim
import numpy as np
import optuna
import polars as pl
import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import umap
from gensim.models.doc2vec import Doc2Vec as Document2Vector
from gensim.models.doc2vec import TaggedDocument
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.decomposition import NMF as NMFModel
from sklearn.decomposition import PCA as PCAModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from transformers import BioGptForCausalLM, BioGptTokenizer

from .utils import ohe_existing_codes

log = logging.getLogger(__name__)


def configure_gensim_logging():
    gensim.models.word2vec.logger.level = logging.WARNING
    gensim.models.doc2vec.logger.level = logging.WARNING
    gensim.utils.logger.level = logging.WARNING


def clean_input_from_unseen(X: Iterable, seen_codes: Iterable[str]) -> np.ndarray:
    seen_codes = set(seen_codes)
    seen_input = []

    for code in X:
        if code in seen_codes:
            seen_input.append(code)

    return np.array(seen_input)


class AutoEncoder(L.LightningModule):
    use_descriptions = False
    text = False

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        df_ohe_lookup: pl.DataFrame,
        learning_rate: float = 0.003,
        hidden_dims=[512, 256, 128],
        weight_decay=1e-4,
        **kwargs,
    ):
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.df_ohe_lookup = df_ohe_lookup
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()

        dropout_rate = 0.5

        # Encoder and Decoder architecture
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

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs = batch
        outputs = self(inputs)

        loss = self.criterion(outputs, inputs)
        self.train_loss(loss.detach())
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch
        outputs = self(inputs)

        loss = self.criterion(outputs, inputs)
        self.val_loss(loss.detach())
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = {
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "min"
            ),  # Creates the ReduceLROnPlateau scheduler
            "monitor": "train/loss_epoch",
            # "monitor": "val/loss",  # The metric you want to monitor (e.g., validation loss)
        }
        return [optimizer], [scheduler]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)

    def transform(self, X: pl.Series):
        self.eval()

        X = ohe_existing_codes(X, self.df_ohe_lookup)

        if np.sum(X) == 0:
            return np.nan

        X = torch.Tensor(X)

        with torch.no_grad():
            embedding = self.encoder(X)

        self.train()

        return embedding.numpy()


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


# TODO: Fix to use polars
# class BioGPT:
#     use_descriptions = True
#     text = True

#     def __init__(
#         self,
#         input_dim: int,
#         embedding_dim: int,
#         df_codes: pl.DataFrame,
#         use_internal_prediction: bool,
#         cache_dir="local/cached/",
#         **kwargs,
#     ):
#         super(BioGPT, self).__init__()
#         self.input_dim = input_dim
#         self.embedding_dim = embedding_dim
#         self.df_codes = df_codes
#         self.use_internal_prediction = use_internal_prediction
#         self.cache_dir = cache_dir
#         self.tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
#         self.model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")

#     def fit(self, _):
#         cache_path = os.path.join(self.cache_dir, "biogpt.pkl")

#         if not os.path.exists(cache_path):
#             log.info("NO BIOGPT CACHE FOUND")
#             embeddings = {"code": [], "embedding": [], "description": []}

#             for row in self.df_codes.itertuples():
#                 if self.use_descriptions:
#                     X = row.description
#                 else:
#                     X = row.code

#                 encoded_input = self.tokenizer(X, return_tensors="pt")
#                 with torch.no_grad():
#                     output = self.model(**encoded_input)

#                 sentence_embeddings = mean_pooling(output, encoded_input["attention_mask"])[0].numpy(force=True)

#                 embeddings["code"].append(row.code)
#                 embeddings["description"].append(row.description)
#                 embeddings["embedding"].append(sentence_embeddings)

#             self.df_codes = pl.DataFrame(embeddings)
#             log.info(f"SAVING BIOGPT CACHE TO {cache_path}")
#             self.df_codes.to_pickle(cache_path)
#         else:
#             log.info(f"FOUND BIOGPT CACHE AT {cache_path}")
#             self.df_codes = pl.read_pickle(cache_path)

#         self.umap_ = UMAPModel(n_components=self.embedding_dim)
#         reduced_embeddings = self.umap_.fit_transform([x for x in self.df_codes["embedding"]])

#         self.df_codes["embedding"] = [x for x in reduced_embeddings]

#     def transform(self, X: Iterable):
#         if isinstance(X, str) or ((isinstance(X, np.ndarray) or isinstance(X, list)) and len(X) == 1):
#             if len(X) == 1:
#                 X = X[0]

#             embedding = self.df_codes.loc[self.df_codes["code"] == X]["embedding"]

#             if len(embedding) == 1:
#                 return embedding.item()
#             else:
#                 return np.nan

#         if self.use_internal_prediction:
#             if isinstance(X, pl.Series):
#                 X = [x for x in X]

#             description = ". ".join(self.df_codes.loc[self.df_codes["code"].isin(X)]["description"])

#             encoded_input = self.tokenizer(description, return_tensors="pt")
#             with torch.no_grad():
#                 output = self.model(**encoded_input)

#             embedding = mean_pooling(output, encoded_input["attention_mask"])[0].numpy(force=True)

#             embedding = self.umap_.transform([embedding])[0]

#             return embedding


# TODO: Fix to use polars
# class Doc2Vec:
#     use_descriptions = False
#     text = True

#     def __init__(
#         self,
#         input_dim: int,
#         embedding_dim: int,
#         window: int = 50,
#         min_count: int = 3,
#         alpha: float = 0.03,
#         dm: int = 1,
#         hs: int = 0,
#         negative: int = 13,
#         epochs: int = 64,
#         workers: int = -1,
#         **kwargs,
#     ):
#         configure_gensim_logging()

#         self.workers = workers
#         if self.workers == -1:
#             self.workers = os.cpu_count()

#         self.embedder = Document2Vector(
#             vector_size=embedding_dim,
#             min_count=min_count,
#             alpha=alpha,
#             dm=dm,
#             hs=hs,
#             negative=negative,
#             epochs=epochs,
#             workers=self.workers,
#         )

#     def fit(self, X: Iterable, ids: Iterable = None):
#         if ids is None:
#             documents_train = [TaggedDocument(doc, [id]) for id, doc in enumerate(X)]
#         else:
#             documents_train = [TaggedDocument(doc, [id]) for id, doc in zip(ids, X)]

#         self.embedder.build_vocab(documents_train)
#         self.embedder.train(documents_train, total_examples=self.embedder.corpus_count, epochs=self.embedder.epochs)

#     def _infer_vector(self, X: Iterable):
#         if isinstance(X, str):
#             X = [X]
#         elif isinstance(X, pl.Series):
#             X = [x for x in X]

#         embeddable_codes = []
#         for x in X:
#             if self.embedder.wv.__contains__(x):
#                 embeddable_codes.append(x)

#         if len(embeddable_codes) > 0:
#             return self.embedder.infer_vector(embeddable_codes, epochs=self.embedder.epochs)
#         else:
#             return np.nan

#     def transform(self, X: Iterable):
#         return self._infer_vector(X)

#     def save(self, path: str):
#         self.embedder.save(path)

#     @classmethod
#     def load(path: str):
#         embedder = Document2Vector.load(path)
#         instance = Doc2Vec.__new__(Doc2Vec)
#         instance.embedder = embedder

#         return instance

#     # @classmethod
#     def hyperopt(self, df: pl.DataFrame, input_dim: int, embedding_dim: int, timeout: int, seed: int, **kwargs) -> dict:
#         n_folds = df["fold"].max()

#         def Objective(trial: optuna.Trial):
#             param = {
#                 "window": trial.suggest_int("window", 1, 32),
#                 "min_count": trial.suggest_int("min_count", 1, 64),
#                 # "epochs": trial.suggest_int("epochs", 8, 64),
#                 "alpha": trial.suggest_float("alpha", 0.001, 0.1),
#                 "dm": trial.suggest_categorical("dm", [0, 1]),
#                 "hs": trial.suggest_categorical("hs", [0, 1]),
#                 # "sample": trial.suggest_float("sample", 0, 1e-5),
#                 "negative": trial.suggest_int("negative", 5, 20),
#             }

#             if param["hs"] == 1:
#                 param["negative"] = 0

#             auc_scores = []

#             for fold in range(n_folds):
#                 df_train = df.loc[df["fold"] != fold].copy()
#                 df_val = df.loc[df["fold"] == fold].copy()

#                 embedder = Doc2Vec(
#                     input_dim=input_dim,
#                     embedding_dim=embedding_dim,
#                     **param,
#                     epochs=self.embedder.epochs,
#                     workers=self.workers,
#                 )
#                 embedder.fit(df_train["codes"])

#                 df_train["embeddings"] = df_train["codes"].apply(embedder.transform)
#                 df_val["embeddings"] = df_val["codes"].apply(embedder.transform)

#                 clf = LogisticRegression(n_jobs=self.workers)
#                 clf.fit([x for x in df_train["embeddings"]], df_train["died"])
#                 probas_fold = clf.predict_proba([x for x in df_val["embeddings"]])[:, 1]

#                 auc_scores.append(roc_auc_score(df_val["died"], probas_fold))

#             score = np.mean(auc_scores)

#             return score

#         sampler = optuna.samplers.TPESampler(seed=seed)

#         study = optuna.create_study(
#             sampler=sampler,
#             direction="maximize",
#             study_name="Doc2Vec optimization",
#         )

#         study.optimize(Objective, gc_after_trial=True, timeout=timeout)

#         hyperopt_score = study.best_value
#         best_params = study.best_params

#         return hyperopt_score, best_params


def create_co_occur_matrix(X: np.array):
    n_codes = X.shape[1]
    co_occur_matrix = np.zeros((n_codes, n_codes), dtype=int)

    for row_codes in X:
        codes = np.where(row_codes == 1)
        for a, b in combinations(codes, 2):
            co_occur_matrix[a, b] += 1
            co_occur_matrix[b, a] += 1  # Because the matrix is symmetric

    return co_occur_matrix


# TODO: Fix to use polars
# class NMF:
#     use_descriptions = False
#     text = False

#     def __init__(
#         self,
#         embedding_dim: int,
#         unique_icd_codes,
#         **kwargs,
#     ):
#         self.embedding_dim = embedding_dim
#         self.unique_icd_codes = unique_icd_codes
#         self.nmf_model = NMFModel(n_components=self.embedding_dim, init="random", random_state=0)

#     def fit(self, X: Iterable):
#         co_occurrence = X.T @ X
#         np.fill_diagonal(co_occurrence, 0)

#         W = self.nmf_model.fit_transform(co_occurrence)
#         self.embedding_matrix = W

#     def transform(self, X: Iterable):
#         if isinstance(X, str):
#             X = [[X]]
#         elif isinstance(X, pl.Series):
#             X = [x for x in X]

#         X = np.array(X)

#         if len(X.shape) == 1:
#             X = np.array([X])

#         X = ohe_with_existing_categories(X, self.unique_icd_codes)

#         if np.sum(X) == 0:
#             return np.nan

#         embedding = X.dot(self.embedding_matrix)

#         return embedding[0]


# TODO: Fix to use polars
# class SVD:
#     use_descriptions = False
#     text = False

#     def __init__(
#         self,
#         embedding_dim: int,
#         unique_icd_codes,
#         **kwargs,
#     ):
#         self.embedding_dim = embedding_dim
#         self.unique_icd_codes = unique_icd_codes

#     def fit(self, X):
#         if not isinstance(X, csr_matrix):
#             X = csr_matrix(X)

#         X = X.astype(float)

#         co_occurrence = X.T.dot(X)
#         co_occurrence.setdiag(0)

#         U, s, Vt = svds(co_occurrence, k=self.embedding_dim)

#         self.vt_reduced = Vt[: self.embedding_dim, :]

#     # def fit(self, X: Iterable):
#     #     co_occurrence = X.T @ X
#     #     np.fill_diagonal(co_occurrence, 0)

#     #     U, s, Vt = np.linalg.svd(co_occurrence, full_matrices=False)

#     #     self.embedding_dim = min(self.embedding_dim, Vt.shape[0])

#     #     self.vt_reduced = Vt[: self.embedding_dim, :]

#     def transform(self, X: Iterable):
#         if isinstance(X, str):
#             X = [[X]]
#         elif isinstance(X, pl.Series):
#             X = [x for x in X]

#         X = np.array(X)

#         if len(X.shape) == 1:
#             X = np.array([X])

#         X = ohe_with_existing_categories(X, self.unique_icd_codes)

#         if np.sum(X) == 0:
#             return np.nan

#         embedding = X.dot(self.vt_reduced.T)

#         return embedding[0]


# TODO: Fix to use polars
# class PCA:
#     use_descriptions = False
#     text = False

#     def __init__(self, embedding_dim: int, unique_icd_codes, **kwargs):
#         self.embedding_dim = embedding_dim
#         self.unique_icd_codes = unique_icd_codes
#         self.pca = PCAModel(n_components=self.embedding_dim)

#     def fit(self, X):
#         co_occurrence = X.T @ X
#         np.fill_diagonal(co_occurrence, 0)

#         self.pca.fit(co_occurrence)

#     def transform(self, X):
#         if isinstance(X, str):
#             X = [[X]]
#         elif isinstance(X, pl.Series):
#             X = [x for x in X]

#         X = np.array(X)

#         if len(X.shape) == 1:
#             X = np.array([X])

#         X = ohe_with_existing_categories(X, self.unique_icd_codes)

#         if np.sum(X) == 0:
#             return np.nan

#         embedding = self.pca.transform(X)

#         return embedding[0]


# TODO: Fix to use polars
# class UMAP:
#     use_descriptions = False
#     text = False

#     def __init__(self, embedding_dim: int, unique_icd_codes, **kwargs):
#         self.embedding_dim = embedding_dim
#         self.unique_icd_codes = unique_icd_codes
#         self.umap = UMAPModel(n_components=self.embedding_dim)

#     def fit(self, X):
#         co_occurrence = X.T @ X
#         np.fill_diagonal(co_occurrence, 0)

#         self.umap.fit(co_occurrence)

#     def transform(self, X):
#         if isinstance(X, str):
#             X = [[X]]
#         elif isinstance(X, pl.Series):
#             X = [x for x in X]

#         X = np.array(X)

#         if len(X.shape) == 1:
#             X = np.array([X])

#         X = ohe_with_existing_categories(X, self.unique_icd_codes)

#         if np.sum(X) == 0:
#             return np.nan

#         embedding = self.umap.transform(X)

#         return embedding[0]


# TODO: Fix to use polars
# class RandomEmbedder:
#     use_descriptions = False
#     text = False

#     def __init__(
#         self,
#         embedding_dim: int,
#         df_codes: pl.DataFrame,
#         **kwargs,
#     ):
#         self.embedding_dim = embedding_dim
#         self.df_codes = df_codes

#     def fit(self, X: Iterable):
#         self.df_codes["embedding"] = self.df_codes["code"].apply(lambda x: np.random.rand(self.embedding_dim))

#     def transform(self, X: Iterable):
#         if isinstance(X, str):
#             X = [X]
#         elif isinstance(X, pl.Series):
#             X = [x for x in X]

#         embedding = self.df_codes.loc[self.df_codes["code"].isin(X)]["embedding"]

#         if len(embedding) >= 1:
#             return embedding.mean()
#         else:
#             return np.nan
