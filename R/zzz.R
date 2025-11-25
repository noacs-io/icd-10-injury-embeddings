.onLoad <- function(libname, pkgname) {
  data("supported_icd_10_codes", package = pkgname, envir = environment())
  data("icd_10_international_codes", package = pkgname, envir = environment())
}
