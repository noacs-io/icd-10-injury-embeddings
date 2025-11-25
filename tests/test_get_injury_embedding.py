import numpy as np
import pytest
from icd_10_injury_embeddings import get_injury_embedding


@pytest.mark.parametrize("dim", [1, 2, 4, 8, 16, 32])
def test_get_injury_embedding_returns_reasonable_values(dim: int) -> None:
    """
    Test embedding generation for various dimensions, ensuring structural integrity,
    numerical validity, and batch consistency.
    """
    # 1. Single patient case
    codes_single = ["S065", "S066"]
    single = get_injury_embedding(codes_single, dim=dim)

    # Type and Shape assertions
    assert isinstance(single, np.ndarray)
    assert single.shape == (dim,)
    assert single.dtype == np.float32

    # Numerical assertions
    assert np.all(np.isfinite(single)), "Embeddings contained NaNs or Infs"
    assert not np.all(single == 0), "Embeddings were all zero"

    # 2. Multiple patients case (Batch)
    codes_multi = [
        ["S065", "S066"],
        ["S270", "S2241", "S271"],
    ]
    multi = get_injury_embedding(codes_multi, dim=dim)

    # Shape assertions for batch
    assert isinstance(multi, np.ndarray)
    assert multi.shape == (2, dim)
    assert np.all(np.isfinite(multi))

    # 3. Reproducibility
    np.testing.assert_allclose(
        multi[0],
        single,
        rtol=1e-5,
        atol=1e-4,
        err_msg="Batch inference yielded different results from single inference",
    )
