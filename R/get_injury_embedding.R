#' Get the injury embedding based from a set of ICD 10 codes.
#'
#' @import dplyr
#' @import purrr
#' @import torch
#'
#' @param icd_10_codes Character vector of ICD-10 codes (without dots, space-separated).
#' @param dim Embedding dimension, one of 1, 2, 4, 16 or 32. Defaults to 8.
#' @param batch_size Rows processed per forward pass. Larger values increase memory use.
#'
#' @return
#' A list of length `length(icd_10_codes)`. Each element is either:
#' \itemize{
#'   \item A numeric vector of length `dim`.
#'   \item `NA` if the patient supplied no valid/mappable ICD-10 codes.
#' }
#'
#' @export
get_injury_embedding <- function(icd_10_codes, dim = 8, batch_size = 409600) {
  # Dynamically select the correct embeddings encoder based on `dim`
  model_file_path <- system.file(paste0("data/r_icd_10_code_encoder_", dim, ".pt"), package = "icd10InjuryEmbeddings")
  encoder <- torch::torch_load(model_file_path)
  encoder$eval()
  
  supported_icd_10_codes <- get("supported_icd_10_codes")
  code_lookup <- split(supported_icd_10_codes$index, supported_icd_10_codes$icd_10_code)
  
  icd_codes_splitted <- strsplit(icd_10_codes, " ")
  
  one_hot_encoded_matrix <- matrix(0, nrow = length(icd_codes_splitted), ncol = nrow(supported_icd_10_codes))
  
  missing_icd_codes <- c()
  
  for (row_index in 1:length(icd_codes_splitted)) {
    for (code in icd_codes_splitted[[row_index]]) {
      if (code %in% names(code_lookup)) {
        col_index <- code_lookup[[code]] + 1
        one_hot_encoded_matrix[row_index, col_index] <- 1
      } else {
        missing_icd_codes <- append(missing_icd_codes, code)
      }
    }
  }
  
  if(length(missing_icd_codes) > 0) {
    missing_icd_codes <- unique(missing_icd_codes)
    warning(sprintf("Could not embed ICD code(s): %s. Refer to \"icdEmbeddings::supported_icd_trauma_codes\" to see embeddable ICD codes.", paste(dQuote(missing_icd_codes), collapse= ", ")))
  }
  
  input <- torch::torch_tensor(one_hot_encoded_matrix, dtype = torch::torch_float32())
  
  # Calculate the number of batches
  n <- dim(input)[1]
  num_batches <- ceiling(n / batch_size)
  
  embeddings <- list()
  
  for(batch_num in 1:num_batches) {
    # Calculate start and end indices for the current batch
    start_idx <- (batch_num - 1) * batch_size + 1
    end_idx <- min(batch_num * batch_size, n)
    
    # Get embeddings for the current batch without gradient calculation
    batch_embeddings <- torch::with_no_grad({
      encoder(input[start_idx:end_idx,])
    }) %>% as.matrix()
    
    for(i in 1:nrow(batch_embeddings)) {
      embedding_index <- (batch_num - 1) * batch_size + i
      
      if(sum(one_hot_encoded_matrix[embedding_index,]) == 0){
        embeddings[[embedding_index]] <- NA
      } else {
        embeddings[[embedding_index]] <- batch_embeddings[i,]
      }
    }
  }
  
  
  return(embeddings)  
}
