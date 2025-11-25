import importlib.resources
from collections.abc import Sequence

import polars as pl

"""
Maps ICD-10-CM codes to their closest ICD-10 (WHO) parent codes.

Typical usage:
    >>> cm_codes = ["S06.0X0A", "S72001A", "s32.401a"]
    >>> icd_10_cm_to_international(cm_codes)
    ['S060', 'S7200', 'S3240']
"""

_DATA_DIR = importlib.resources.files("icd_10_injury_embeddings") / "data"
df_icd_10_international_codes = pl.read_csv(_DATA_DIR / "icd_10_international_codes.csv")


def icd_10_cm_to_international(icd_10_cm_codes: Sequence[str]) -> list[str]:
    """
    Convert ICD-10-CM codes to their closest ICD-10 (WHO) ancestor by progressively trimming
    the suffix until a match appears in the reference catalogue or the code reaches three characters.

    Args:
        icd_10_cm_codes: Iterable of ICD-10-CM strings (e.g. ["S06.0X0A", "S72001A"]).

    Returns:
        List of mapped ICD-10 (WHO) codes, preserving the input order.
    """
    reference_codes = set(df_icd_10_international_codes["code"].to_list())

    cleaned_codes = []
    for code in icd_10_cm_codes:
        text = str(code).upper().replace(".", "")
        cleaned_codes.append(text)

    mapping = {}
    for code in set(cleaned_codes):
        current = code
        while len(current) > 3 and current not in reference_codes:
            current = current[:-1]
        mapping[code] = current

    return [mapping[code] for code in cleaned_codes]
