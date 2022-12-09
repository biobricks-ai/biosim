reticulate::use_condaenv("deepchem")
pacman::p_load(keras,tensorflow,tidyverse,tfdatasets,RSQLite)

# GET CHEMBERT FROM HUGGINGFACE =================================================
transformers <- reticulate::import('transformers') 
hface <- "seyonec/ChemBERTa-zinc-base-v1"
tok   <- transformers$AutoTokenizer$from_pretrained(hface)
bert  <- transformers$TFAutoModel$from_pretrained(hface, from_pt=TRUE)

embed_bert <- \(smi){
  ids <- 1:length(smi) - 1
  res <- tok$batch_encode_plus(smi,padding="max_length", max_length=512L)  
  enc <- do.call(rbind,map(ids, ~ res[.]$ids))
  att <- do.call(rbind,map(ids, ~ res[.]$attention_mask))
  tokens <- list(input_ids=enc, attention_mask=att)
  array(bert$predict(tokens)[[2]],dim=c(length(ids),768L))
}