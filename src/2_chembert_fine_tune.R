# use the conda environment created with conda env create --name deepchem python=3.9
reticulate::use_condaenv("deepchem")
pacman::p_load(keras,tensorflow,tidyverse)

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
  mutate(value = array(ifelse(standard_value>0,1L,0L))) |>
  sample_n(1000)

tokens <- mktokens(data$canonical_smiles) |> map(~ keras_array(.,dtype="int32"))
embeds <- array(bert$predict(tokens)[[2]],dim=c(nrow(data),768L))
embeds <- tf$linalg$normalize(embeds,axis=-1L)[[1]]

train_idx <- 1:ceiling(nrow(data)*0.9)
test_idx <- setdiff(1:nrow(data),train_idx)
train <- list(x = embeds[train_idx,], y = data$value[train_idx])
test  <- list(x = embeds[test_idx,],  y = data$value[test_idx])

build_model <- function(){
  iput <- layer_input(shape = c(768), dtype = "float")
  hidden <- iput |> layer_dense(units = 100, activation="relu")
  hnorm <- hidden |> layer_layer_normalization()
  oput <- hnorm |> layer_dense(units = 1, activation="sigmoid")
  keras_model(iput, oput)
}

mod <- build_model()
mod$compile(optimizer = optimizer_adam(), loss = "binary_crossentropy", metrics = "accuracy")

mod$fit(train$x, train$y, epochs = 100L, batch_size = 32L, 
  validation_data = list(test$x, test$y))
mod$evaluate(test$x, test$y)

get_activation <- function(layer,instance){
  keras_model(inputs = mod$inputs, outputs = layer$output) |> predict(instance)
}

# EVALUATION ===========================================================
activation = get_activation(mod$layers[[2]],embeds) 
activation = tf$linalg$normalize(activation,axis=-1L)[[1]]

sim <- tf$matmul(activation,tf$transpose(activation)) |> as.matrix()
esim <- tf$matmul(embeds,tf$transpose(embeds)) |> as.matrix()

simdata <- tibble(left=data$canonical_smiles,leftval=data$value) |>
  full_join(tibble(right=data$canonical_smiles,rightval=data$value),by=character()) |>
  mutate(sup_emb = array(sim,dim=nrow(data)^2)) |>
  mutate(brt_emb = array(esim,dim=nrow(data)^2)) |>
  mutate(x = ceiling(row_number()/nrow(data))) |>
  mutate(y = row_number()-(x-1)*nrow(sim)) |>
  mutate(set = ifelse(x %in% test_idx | y %in% test_idx,"test","train"))

## CLUSTERING HEATMAPS
(function(){ 
  jpeg(file="heatmap.activation.jpg"); heatmap(sim); dev.off()
  jpeg(file="heatmap.embedding.jpg"); heatmap(esim); dev.off()
})()

## ACCURACY EVALUATION
simdata |> 
  mutate(same_val = leftval == rightval) |>
  tidyr::pivot_longer(5:6,names_to="embedding",values_to="sim") |>
  filter(sim > 0.75) |> 
  count(embedding,set,same_val)

# find examples where compounds are 
# 1. similar according to embedding
# 2. not similar according to activation
eg <- simdata |> filter(brt_emb > 0.9, sup_emb < 0.5, x < y) |>
  select(left,right,leftval,rightval,brt_emb,sup_emb) |>
  arrange(desc(brt_emb)) |>
  head(10)  

smiles <- map(1:nrow(eg), \(i){ list(eg$left[i],eg$right[i]) }) |> 
  flatten() |> as.character()

res <- callr::r_bg(function(smiles){
  reticulate::use_condaenv("r-reticulate")
  reticulate::source_python('src/py/draw_mols.py')
  draw_mols(smiles,"output2.svg")
},args=list(smiles=smiles))

# similar according to activation, but not according to embedding
eg <- simdata |> filter(x < y,sup_emb>0.8) |>
  mutate(dif = sup_emb - brt_emb) |>
  arrange(desc(dif)) |>
  head(10)  

smiles <- map(1:nrow(eg), \(i){ list(eg$left[i],eg$right[i]) }) |> 
  flatten() |> as.character()

res <- callr::r_bg(function(smiles){
  reticulate::use_condaenv("r-reticulate")
  reticulate::source_python('src/py/draw_mols.py')
  draw_mols(smiles,"output2.svg")
},args=list(smiles=smiles))

