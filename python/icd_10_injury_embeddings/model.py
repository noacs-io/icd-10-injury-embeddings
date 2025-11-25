"""
Provides pre-trained autoencoder models for generating dense vector representations
of injury patterns from ICD-10 codes.

Typical usage:
    >>> # Single patient with multiple injury codes
    >>> codes = ["S065", "S066"]
    >>> embedding = get_injury_embedding(codes, dim=16)
    >>> embedding.shape
    (16,)

    >>> # Batch of patients
    >>> batch_codes = [["S065", "S066"], ["S270", "S2241", "S271"]]
    >>> embeddings = get_injury_embedding(batch_codes, dim=8)
    >>> embeddings.shape
    (2, 8)
"""

import importlib.resources
import warnings

import numpy as np
import polars as pl
import torch
import torch.nn as nn

_DATA_DIR = importlib.resources.files("icd_10_injury_embeddings") / "data"

df_supported_icd_10_codes = pl.read_csv(_DATA_DIR / "supported_icd_10_codes.csv").rename({"index": "ohe_index"})


class TorchAutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dims: list[int] | None = None,
        dropout_rate: float = 0.5,
    ):
        """
        Args:
            input_dim: Dimension of one-hot encoded input space (806 supported codes)
            embedding_dim: Target dimension for compressed representation
            hidden_dims: Three-element list specifying intermediate layer sizes.
                Defaults to [512, 256, 128] for symmetric compression.
            dropout_rate: Dropout probability for regularisation (default 0.5)
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, input_dim)

        Returns:
            Tensor of shape (batch, input_dim)
        """
        return self.decoder(self.encoder(x))


def load_auto_encoder(embedding_dim: int) -> nn.Sequential:
    """Load pre-trained encoder weights.

    Args:
        embedding_dim: Dimension of embedding to load

    Returns:
        Initialised encoder module with frozen pre-trained weights
    """
    model = TorchAutoEncoder(input_dim=806, embedding_dim=embedding_dim)
    state_path = _DATA_DIR / f"icd_10_code_encoder_{embedding_dim}.pt"
    state_dict = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model.encoder


def one_hot_encode_icd_10_codes(icd_10_codes: list[str] | list[list[str]]) -> np.ndarray:
    """Convert ICD-10 strings to sparse binary matrix using a set indexing.

    Handles both single patient (1D) and batched (2D) inputs.
    Filters unsupported codes and generates one-hot representation over the
    supported code vocabulary.

    Args:
        icd_10_codes: List of ICD-10 codes. Can be:
            - 1D: Single patient list, e.g. ["S065", "S066"]
            - 2D: List of patient code lists, e.g. [["S065", "S066"], ["S270", "S2241", "S271"]]

    Returns:
        Binary array of shape (n_patients, 806) for 2D input or (806,) for 1D input
        where each row/vector indicates presence of supported codes
    """
    if not icd_10_codes:
        return np.zeros((0, len(df_supported_icd_10_codes)), dtype=np.int8)

    is_1d = isinstance(icd_10_codes[0], str)
    codes_2d = [icd_10_codes] if is_1d else icd_10_codes

    df_codes = (
        pl.DataFrame({"icd_10_code": codes_2d}, schema={"icd_10_code": pl.List(pl.String)})
        .with_row_index()
        .explode("icd_10_code")
    )

    missing_codes = df_codes.filter(
        (pl.col("icd_10_code").is_not_null())
        & (~pl.col("icd_10_code").is_in(df_supported_icd_10_codes["icd_10_code"].implode()))
    )["icd_10_code"].unique()

    if not missing_codes.is_empty():
        warnings.warn(
            f"Could not embed ICD code(s): {', '.join(missing_codes.to_list())}. "
            f"Refer to 'supported_icd_10_codes' (n={len(df_supported_icd_10_codes)}) "
            "for embeddable ICD codes.",
            UserWarning,
            stacklevel=2,
        )

    df_codes = df_codes.filter(
        (pl.col("icd_10_code").is_not_null())
        & (pl.col("icd_10_code").is_in(df_supported_icd_10_codes["icd_10_code"].implode()))
    )

    df_codes = df_codes.join(df_supported_icd_10_codes, on="icd_10_code")

    one_hot_matrix = np.zeros((len(codes_2d), len(df_supported_icd_10_codes)), dtype=np.int8)

    row_col_pairs = df_codes.select(["index", "ohe_index"]).to_numpy()
    one_hot_matrix[row_col_pairs[:, 0].astype(int), row_col_pairs[:, 1].astype(int)] = 1

    return one_hot_matrix.squeeze() if is_1d else one_hot_matrix


def get_injury_embedding(icd_10_codes: list[str] | list[list[str]], dim: int = 8) -> np.ndarray:
    """Generate dense embeddings for patient injury patterns.

    Handles both single-patient and batched inputs. Unsupported codes are
    filtered prior to embedding generation. Returns np.nan vectors for patients
    with only unsupported codes.

    Args:
        icd_10_codes: Single list of codes (e.g. ["S065", "S066"]) or batch
            of patient code lists (e.g. [["S065", "S066"], ["S270", "S2241", "S271"]])
        dim: Embedding dimension (8, 16, 32, or 64 depending on trained models)

    Returns:
        For single-patient input: 1D array of shape (dim,) filled with np.nan if all codes missing
        For batched input: 2D array of shape (n_patients, dim) with np.nan rows for patients with all missing codes

    Example:
        >>> # Trauma patient with multiple injuries
        >>> get_injury_embedding(["S065", "S066"], dim=16)
        array([...], dtype=float32)
    """
    model = load_auto_encoder(dim)

    is_single_patient = False
    if icd_10_codes and isinstance(icd_10_codes[0], str):
        is_single_patient = True
        batch_input = [icd_10_codes]
    else:
        batch_input = icd_10_codes

    one_hot_encoded_matrix = one_hot_encode_icd_10_codes(batch_input)

    if len(batch_input) == 0:
        return np.full((dim), np.nan, dtype=np.float32)

    zero_rows = np.all(one_hot_encoded_matrix == 0, axis=1)

    embeddings = np.full((len(batch_input), dim), np.nan, dtype=np.float32)

    if not np.all(zero_rows):
        valid_mask = ~zero_rows
        valid_tensor = torch.tensor(one_hot_encoded_matrix[valid_mask], dtype=torch.float32)
        with torch.no_grad():
            valid_embeddings = model(valid_tensor).numpy()
        embeddings[valid_mask] = valid_embeddings

    return embeddings[0] if is_single_patient else embeddings
