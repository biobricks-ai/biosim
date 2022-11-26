# Setup --------------------------------------------------------------------
pacman::p_load(keras,reticulate,tensorflow,tidyverse)
reticulate::use_condaenv("deepchem")

# Data preparation --------------------------------------------------------
# TODO CDDD, pubchem
# TODO learn embedding for each assay
# TODO build directional field for each assay

data <- readr::read_csv("cache/train.csv") |>
  filter(stype=="1998110-%-Percent Effect") |>
  mutate(value = ifelse(standard_value>0,1,0))

tokenizer <- text_tokenizer(31,lower=F,char_level=T,filters='') 
fit_text_tokenizer(tokenizer,data$canonical_smiles[1:1e4])

padsize   <- 200  
vectorize <- \(v){ v[1:padsize] |> modify_if(is.na,~ as.integer(-1)) }
data$seq <- tokenizer$texts_to_sequences(data$canonical_smiles)
data$seq <- pbapply::pblapply(data$seq, vectorize)

reticulate::source_python('src/py/save_tfrec.py')
fs::dir_create("cache/tfrecord/structure.tfrecord")
fs::dir_create("cache/tfrecord/label.tfrecord")

fs::dir_ls("cache/tfrecord/structure.tfrecord") |> fs::file_delete()
fs::dir_ls("cache/tfrecord/label.tfrecord") |> fs::file_delete()
halfbaked::chunk(1:nrow(data),10000) |> iwalk(\(chunk,i){
  chunk <- data[chunk,]
  X <- tf$one_hot(chunk$seq,depth=31L)
  Y <- tf$one_hot(array(chunk$value,dim=c(nrow(chunk))),depth=2L)
  
  write_tensor(X, glue::glue("cache/tfrecord/structure.tfrecord/structure_{i}.tfrecord"))
  write_tensor(Y, glue::glue("cache/tfrecord/label.tfrecord/label_{i}.tfrecord"))
})

X <- tfdatasets::file_list_dataset("cache/tfrecord/structure.tfrecord")


# Train it --------------------------------------------------------
train <- function(){
  # With TF-2, you can still run this code due to the following line:
  if(tf$executing_eagerly()){ tf$compat$v1$disable_eager_execution() }
  K <- keras::backend()
  source('src/models/cvae.R')
}


# Test it --------------------------------------------------------
encoder <- keras::load_model_hdf5("brick/encoder.hdf5")
encoded <- predict(encoder, X3)
res <- tibble(smi = data$canonical_smiles, enc=encoded) |> distinct()

norm <- \(x){ sum(sqrt(x^2)) }
res$mag <- map_dbl(1:nrow(res),~ norm(res$enc[.,]))
res <- res |> mutate(nenc = enc/mag)

anc <- res[1:100,] |> select(anc=smi,enc_a=enc)
ana <- res[101:nrow(res),] |> select(ana=smi, enc_b=enc)

library(lsa)
res2 <- anc |> dplyr::full_join(ana,by=character()) 
res2 <- res2 |> mutate(sim = pbapply::pbsapply(1:nrow(res2),\(i){ lsa::cosine(res2$enc_a[i,],res2$enc_b[i,])} ))
saveRDS(res2,"similarity.rds")

res2 <- readRDS("similarity.rds") |> tibble()
reticulate::source_python('src/py/draw_mols.py')

res3 <- res2 |> filter(anc!=ana) |> group_by(anc) |> slice_max(order_by=sim,n=1,with_ties = F) |> ungroup()
mols <- 1:nrow(res3) |> map(\(i){ c(res3[i,"anc"],res3[i,"ana"])}) |> unlist()
draw_mols(mols,"output.svg")
