import pytest
from icd_10_injury_embeddings import icd_10_cm_to_international


def test_icd_10_cm_to_international_cleans_input() -> None:
    cm_input = [
        "S06.0X0A",
        "S72001A",
        "s32.401a",
        "T14.90XA",
        "S01.01XA",
        "S72001A",
    ]
    expected_output = [
        "S060",
        "S7200",
        "S3240",
        "T149",
        "S010",
        "S7200",
    ]

    assert icd_10_cm_to_international(cm_input) == expected_output
