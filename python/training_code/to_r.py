import polars as pl
import torch
from polars import col

from .lib import TorchAutoEncoder
from .utils import handle_missing_codes

EMBEDDING_DIMS = [1, 2, 4, 8, 16, 32]
df_suppored_codes = pl.read_csv("results/supported_codes.csv").rename({"index": "ohe_index"})
df_trauma_codes = pl.read_csv("local/data/all_supported_icd_codes.csv")

for EMBEDDING_DIM in EMBEDDING_DIMS:
    CHECKPOINT_PATH = f"local/checkpoints/{EMBEDDING_DIM}.ckpt"

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device("cpu"))

    torch.save(checkpoint["state_dict"], f"local/weights/{EMBEDDING_DIM}.pt")

    loaded_model = TorchAutoEncoder(
        input_dim=len(df_suppored_codes), embedding_dim=EMBEDDING_DIM, df_ohe_lookup=df_suppored_codes
    )
    # state_dict = torch.load(f"results/weights/{EMBEDDING_DIM}.pt")
    state_dict = torch.load(f"local/weights/{EMBEDDING_DIM}.pt")
    loaded_model.load_state_dict(state_dict)

    loaded_model.eval()

    print("EMBEDDING CODES")
    codes = df_trauma_codes["icd_code"].map_elements(lambda x: [x], pl.List(pl.String))

    embeddings = loaded_model.embed(codes)

    df_trauma_codes = df_trauma_codes.with_columns(pl.lit(embeddings).alias("embedding"))

    if df_trauma_codes["embedding"].null_count() > 0:
        print(f"HANDLING MISSING CODES (MISSING {df_trauma_codes['embedding'].null_count()})")
        df_trauma_codes = handle_missing_codes(df_trauma_codes)

    df_trauma_codes.write_parquet(f"local/embeddings/{EMBEDDING_DIM}.parquet")

    df_trauma_codes = df_trauma_codes.with_columns(
        col("embedding").cast(pl.Array(pl.String, EMBEDDING_DIM)).arr.join(",").alias("embedding")
    )

    df_trauma_codes.write_csv(f"local/embeddings/{EMBEDDING_DIM}.csv")

    print(f"COULD NOT EMBED {df_trauma_codes['embedding'].null_count()} CODES")
