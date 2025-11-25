test_that("get_injury_embedding returns reasonable values", {
  for (dim in c(1L, 2L, 4L, 8L, 16L, 32L)) {
    # single patient
    single <- get_injury_embedding("S065 S066", dim = dim)
    expect_type(single, "list")
    expect_length(single, 1L)
    expect_length(single[[1]], dim)
    expect_type(single[[1]], "double")
    expect_true(all(is.finite(single[[1]])))
    expect_false(all(single[[1]] == 0))

    # multiple patients
    multi <- get_injury_embedding(c("S065 S066", "S270 S2241 S271"), dim = dim)
    expect_length(multi, 2L)
    expect_true(all(lengths(multi) == dim))

    # reproducibility
    expect_equal(multi[[1]], single[[1]], tolerance = 1e-6)
  }
})
