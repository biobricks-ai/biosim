# use the conda environment created with conda env create --name deepchem python=3.9
reticulate::use_condaenv("deepchem")
pacman::p_load(keras,tensorflow,tidyverse)

# Install some packages
# reticulate::py_install("torch", pip = TRUE)
# reticulate::py_install("transformers", pip = TRUE)

# TODO CDDD

transformers <- reticulate::import('transformers') # this is a huggingface.co api

hface <- "seyonec/ChemBERTa-zinc-base-v1"
tok   <- transformers$AutoTokenizer$from_pretrained(hface)
bert  <- transformers$TFAutoModel$from_pretrained(hface, from_pt=TRUE)

mktokens <- \(smi){
  res <- tok$batch_encode_plus(smi,padding="max_length", max_length=512L)
  enc <- do.call(rbind,map(1:length(smi) - 1, \(i){ res[i]$ids})) 
  att <- do.call(rbind,map(1:length(smi) - 1, \(i){ res[i]$attention_mask})) 
  list(input_ids=enc, attention_mask=att)
}

data <- readr::read_csv("cache/train.csv") |>
  filter(stype=="1998110-%-Percent Effect") |>
  mutate(value = ifelse(standard_value>0,1,0)) |>
  sample_n(1000)

tokens <- mktokens(data$canonical_smiles) |> map(~ keras_array(.,dtype="int32"))
embeds <- bert$predict(tokens)[[2]]
norm   <- tf$linalg$normalize(embeds,axis=-1L)[[1]]

sim <- tf$matmul(norm,tf$transpose(norm))

anc <- tibble(anc=data$canonical_smiles)
ana <- tibble(ana=data$canonical_smiles) |> mutate(row=row_number())

hmap <- anc |> full_join(ana,by=character()) |> mutate(sim=array(sim)) |> arrange(-sim) |>
  filter(anc < ana)

smi <- hmap |> pivot_longer(cols=c("anc","ana")) |> filter(sim>0.93,sim<0.99)

res <- callr::r_bg(function(smiles){
  reticulate::use_condaenv("r-reticulate")
  reticulate::source_python('../src/py/draw_mols.py')
  draw_mols(smiles,"output2.svg")
},args=list(smiles=smi$value))


o2 <- bert$layers[[1]] |> layer_dense(units=5)
test <- keras::keras_model()
