reticulate::use_condaenv("deepchem", required = TRUE)
pacman::p_load(biobricks, arrow, tidyverse, magrittr, tfdatasets)

activities <- brick_load("chemharmony")$harmonized$activities.parquet |> collect()

tfds <- tfdatasets::tensor_slices_dataset(list(
  smiles = tf$constant(activities$smiles, dtype=tf$string),
  pid    = tf$constant(activities$pid,    dtype=tf$string),
  value  = tf$constant(activities$value,  dtype=tf$float32)))

tmpdir <- withr::local_tempdir()
batch <- tfds |> tfdatasets::dataset_batch(1000)
bpath <- fs::path(tmpdir,"batches")
batch$save(bpath)

# BUILD EMBEDDINGS ==========================================================
build_activity_embeddings_ds <- function(batchidx,batchpath,outpath){

  reticulate::use_condaenv("deepchem", required = TRUE)
  pacman::p_load(biobricks, arrow, tidyverse, magrittr, tfdatasets)

  rows <- tf$data$Dataset$load(batchpath) |> dataset_skip(batchidx) |> dataset_take(1)
  rows <- reticulate::as_iterator(rows) |> reticulate::iter_next()

  smi <- map_chr(rows$smiles$numpy(), ~ .$decode())

  ## CHEMBERT ===============
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

  bert = tf$constant(embed_bert(smi),dtype=tf$float32)

  ## PUBCHEM2D ==============
  source("src/scala/mol2pubchem.R", chdir = TRUE) # import mol2pubchem function
  smiles2pubchem <- pubchembuilder()
  
  p2d <- map(smi, ~ smiles2pubchem(.)) # pubchem2d vectors
  idx <- which(map_lgl(p2d, ~ length(.) == 881)) # find valid pubchem2d vectors
  p2d[!idx] <- rep(-1,881) # replace invalid pubchem2d vectors with -1

  pubchem <- array(unlist(p2d), dim=c(nrow=length(p2d),ncol=881))
  pubchem <- tf$constant(pubchem,dtype=tf$int32)

  ds = tfdatasets::tensor_slices_dataset(list(
    smiles   = rows$smiles,
    pid      = rows$pid,
    value    = rows$val,
    emb_p2d  = pubchem,
    emb_bert = bert
  ))

  ds$save(fs::dir_create(outpath))
}

system.time({
  build_activity_embeddings_ds(0,bpath,fs::path(tmpdir,"batch0"))
}) # 1.5 min

bg = callr::r_bg(
  build_activity_embeddings_ds, 
  args = list(batchidx=0, batchpath=bpath, outpath=fs::path(tmpdir,"batch0")))

# print standard output from bg
while (bg$is_alive()) {
  cat(bg$read_output_line(), "\n")
}

fs::dir_ls(fs::path(tmpdir,"batch0"))
