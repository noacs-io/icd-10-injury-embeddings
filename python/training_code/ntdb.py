import re
import warnings

import lightgbm
import numpy as np
import polars as pl
import xgboost as xgb
from polars import col
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from .lib import get_patient_embedding_autoencoder, get_patient_embedding_mean, one_hot_encode_icd_codes

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=re.escape("X does not have valid feature names, but LGBMClassifier was fitted with feature name"),
)

EMBEDDING_DIMS = [2, 4, 8, 16, 32]
TARGETS = ["in_hospital_mortality", "ed_disposition", "blood_transfusion_24h"]

# EMBEDDING_DIMS = [2]
# TARGETS = ["ed_disposition"]


# Load data once at the beginning
def load_and_prepare_data():
    df_train = pl.read_csv("local/dataset_2018.csv", null_values=["NA"]).with_columns(pl.lit("TRAIN").alias("set"))
    df_test = pl.read_csv("local/dataset_2019.csv", null_values=["NA"]).with_columns(pl.lit("TEST").alias("set"))

    common_facilities = set(df_train["facility_key"]).intersection(set(df_test["facility_key"]))

    df_train = df_train.filter(pl.col("facility_key").is_in(common_facilities))
    df_test = df_test.filter(pl.col("facility_key").is_in(common_facilities))

    df = pl.concat([df_train, df_test])

    df = df.filter(col("icd_codes").is_not_null())

    return df


def run_prediction_models(X_train, y_train, X_val, y_val):
    n_classes = len(np.unique(y_train))

    if not np.issubdtype(y_train.dtype, np.number):
        le = LabelEncoder()

        y_train = le.fit_transform(y_train)
        y_val = le.transform(y_val)

    if n_classes > 2:
        clf_logit = OneVsRestClassifier(LogisticRegression(n_jobs=-1, max_iter=1000))
        clf_lgb = lightgbm.LGBMClassifier(objective="multiclass", num_class=n_classes, verbose=-1)
    else:
        clf_logit = LogisticRegression(n_jobs=-1, max_iter=1000)
        clf_lgb = lightgbm.LGBMClassifier(objective="binary", verbose=-1)

    clf_logit.fit(X_train, y_train)
    clf_lgb.fit(X_train, y_train)

    probas_logit = clf_logit.predict_proba(X_val)
    probas_lgb = clf_lgb.predict_proba(X_val)

    try:
        if n_classes == 2:
            auc_logit = roc_auc_score(y_val, probas_logit[:, 1])
            auc_lgb = roc_auc_score(y_val, probas_lgb[:, 1])
        else:
            auc_logit = roc_auc_score(y_val, probas_logit, multi_class="ovr", average="macro")
            auc_lgb = roc_auc_score(y_val, probas_lgb, multi_class="ovr", average="macro")
    except:
        return None

    results_logit = pl.DataFrame(
        {
            "model": ["LOGREG"],
            "auc": [float(auc_logit)],
        }
    )

    results_lgb = pl.DataFrame(
        {
            "model": ["LIGHTGBM"],
            "auc": [float(auc_lgb)],
        }
    )

    return pl.concat([results_logit, results_lgb])


def process_facility(df_facility, target, input_data):
    df_facility_train = df_facility.filter(col("set") == "TRAIN")

    df_facility_val = df_facility.filter(col("set") == "TEST")
    # ensure all validation samples have a seen target
    df_facility_val = df_facility_val.filter(col(target).is_in(df_facility_train[target].unique()))

    X_train = df_facility_train[input_data].to_numpy()
    y_train = df_facility_train[target].to_numpy()

    X_val = df_facility_val[input_data].to_numpy()
    y_val = df_facility_val[target].to_numpy()

    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        return None

    res = run_prediction_models(X_train, y_train, X_val, y_val)

    return res


def normalise_embeddings(embeddings):
    min_vals = np.min(embeddings, axis=0)
    max_vals = np.max(embeddings, axis=0)

    normalised = (embeddings - min_vals) / (max_vals - min_vals)

    return normalised


df = load_and_prepare_data()


df_results = pl.DataFrame(
    {},
    schema={
        "target": pl.String,
        "input_data": pl.String,
        "facility_key": pl.Int64,
        "model": pl.String,
        "auc": pl.Float64,
    },
)

facility_keys = df["facility_key"].unique().sort()

for embedding_dim in tqdm(EMBEDDING_DIMS, desc="Embeddings", leave=False):
    embeddings = get_patient_embedding_autoencoder(df["icd_codes"], embedding_dim)
    embeddings = normalise_embeddings(embeddings)

    df_embeddings = df.with_columns(pl.lit(embeddings).alias("embedding")).select(
        TARGETS + ["facility_key", "embedding", "set"]
    )

    for target in tqdm(TARGETS, desc="Target", leave=False):
        df_all = df_embeddings.filter(col(target).is_not_null())

        df_results_all = process_facility(df_all, target, "embedding")

        if df_results_all is None:
            continue

        df_results_all = df_results_all.with_columns(
            pl.lit(-1, dtype=pl.Int64).alias("facility_key"),
            pl.lit(target).alias("target"),
            pl.lit(f"EMBEDDING_{embedding_dim}").alias("input_data"),
        ).select(df_results.columns)

        df_results = pl.concat([df_results, df_results_all])

        for facility_key in tqdm(facility_keys, desc="Facilities", leave=False):
            df_facility = df_embeddings.filter((col("facility_key") == facility_key) & col(target).is_not_null())

            df_results_facility = process_facility(df_facility, target, "embedding")

            if df_results_facility is None:
                continue

            df_facility_results = df_results_facility.with_columns(
                pl.lit(facility_key, dtype=pl.Int64).alias("facility_key"),
                pl.lit(target).alias("target"),
                pl.lit(f"EMBEDDING_{embedding_dim}").alias("input_data"),
            ).select(df_results.columns)

            df_results = pl.concat([df_results, df_facility_results])

for target in tqdm(TARGETS, desc="Target (OHE)", leave=False):
    df_ohe = df.select(TARGETS + ["facility_key", "icd_codes", "set"])

    df_all = df_ohe.filter(col(target).is_not_null())

    df_facility_train = df_all.filter(col("set") == "TRAIN")

    train_icd_codes = df_facility_train["icd_codes"].str.split(" ").explode().unique()
    df_train_icd_codes = pl.DataFrame({"code": train_icd_codes}).with_row_index().rename({"index": "ohe_index"})

    icd_codes_ohe = one_hot_encode_icd_codes(df_all["icd_codes"], df_train_icd_codes)

    df_all = df_all.with_columns(pl.lit(icd_codes_ohe).alias("icd_codes_ohe"))

    df_results_all = process_facility(df_all, target, "icd_codes_ohe")

    if df_results_all is None:
        continue

    df_results_all = df_results_all.with_columns(
        pl.lit(-1, dtype=pl.Int64).alias("facility_key"),
        pl.lit(target).alias("target"),
        pl.lit("ONE_HOT_ENCODING").alias("input_data"),
    ).select(df_results.columns)

    df_results = pl.concat([df_results, df_results_all])

    for facility_key in tqdm(facility_keys, desc="Facilities", leave=False):
        df_facility = df_ohe.filter((col("facility_key") == facility_key) & col(target).is_not_null())

        df_facility_train = df_facility.filter(col("set") == "TRAIN")

        train_icd_codes = df_facility_train["icd_codes"].str.split(" ").explode().unique()
        df_train_icd_codes = pl.DataFrame({"code": train_icd_codes}).with_row_index().rename({"index": "ohe_index"})

        icd_codes_ohe = one_hot_encode_icd_codes(df_facility["icd_codes"], df_train_icd_codes)

        df_facility = df_facility.with_columns(pl.lit(icd_codes_ohe).alias("icd_codes_ohe"))

        df_results_facility = process_facility(df_facility, target, "icd_codes_ohe")

        if df_results_facility is None:
            continue

        df_results_facility = df_results_facility.with_columns(
            pl.lit(facility_key, dtype=pl.Int64).alias("facility_key"),
            pl.lit(target).alias("target"),
            pl.lit("ONE_HOT_ENCODING").alias("input_data"),
        ).select(df_results.columns)

        df_results = pl.concat([df_results, df_results_facility])

df_results.write_csv("results/validation.csv")
