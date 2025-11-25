test_that("icd_10_cm_to_international correctly cleans input", {
  cm_input <- c(
    "S06.0X0A",
    "S72001A", 
    "s32.401a",
    "T14.90XA", 
    "S01.01XA",
    "S72001A" 
  )
  expected_output <- c(
    "S060", 
    "S7200", 
    "S3240", 
    "T149", 
    "S010",  
    "S7200" 
  )

  result <- icd_10_cm_to_international(cm_input)

  expect_equal(result, expected_output)
})
