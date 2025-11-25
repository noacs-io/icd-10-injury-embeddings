#' Map ICD-10-CM to International ICD-10 (WHO)
#'
#' Converts a vector of ICD-10-CM codes to their closest International ICD-10 parent code.
#' It relies on the package data object \code{icd_10_international_codes} as the reference.
#'
#' @param icd_10_cm_codes Character vector of ICD-10-CM codes to convert.
#'
#' @return A character vector of mapped codes (same length and order as input).
#'
#' @import dplyr
#' @importFrom purrr map_chr
#' @export
icd_10_cm_to_international <- function(icd_10_cm_codes) {
  icd_10_international_codes <- get("icd_10_international_codes")$code

  if (!is.character(icd_10_cm_codes)){
    icd_10_cm_codes <- as.character(icd_10_cm_codes)
  } 

  icd_10_cm_codes <- gsub("\\.", "", toupper(icd_10_cm_codes))

  lookup_table <- tibble(original_code = unique(icd_10_cm_codes)) %>%
    mutate(
      mapped_code = purrr::map_chr(original_code, function(code) {

        current <- code
        while (nchar(current) > 3 && !(current %in% icd_10_international_codes)) {
          current <- substr(current, 1, nchar(current) - 1)
        }
        return(current)

      })
    )

  result <- tibble(original_code = icd_10_cm_codes) %>%
    left_join(lookup_table, by = "original_code") %>%
    pull(mapped_code)

  return(result)
}
