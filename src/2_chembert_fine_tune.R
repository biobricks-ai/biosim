# use the conda environment created with conda env create --name deepchem python=3.9
reticulate::use_condaenv("deepchem")
pacman::p_load(keras,tensorflow,tidyverse,tfdatasets,RSQLite)


# fs::dir_ls("cache/tfrecord") |> fs::file_size()

parse_fun <- function(example_proto) {
  features <- dict(
    "smiles"  = tf$io$FixedLenFeature(shape(1), tf$string),
    "embchem" = tf$io$FixedLenFeature(shape(768), tf$float32),
    "embprop" = tf$io$FixedLenFeature(shape(1), tf$int64),
    "output"  = tf$io$FixedLenFeature(shape(1), tf$int64)
  )
  
  feat <- tf$io$parse_single_example(example_proto, features)
  list(list(embchem=feat$embchem,embprop=feat$embprop),feat$output)
}

identity <- reticulate::r_to_py(\(x){ x })

instances_per_file <- 1000 
tfrecs <- fs::dir_ls("cache/sims.tfrecord")
Ndata  <- instances_per_file*length(tfrecs)
data   <- tfrecord_dataset(tfrecs) |> dataset_map(parse_fun) 
train  <- data |> dataset_take(0.8*Ndata) |> dataset_batch(100) |> dataset_repeat()
test   <- data |> dataset_skip(0.8*Ndata) |> dataset_take(0.1*Ndata) |> dataset_batch(100)
vali   <- data |> dataset_skip(0.9*Ndata) |> dataset_take(0.1*Ndata) |> dataset_batch(100)

build_model <- function(){
  keras::k_clear_session()
  embed_smiles <- layer_input(shape = c(768L), dtype = "float", name="embchem") 
  norm_smiles  <- embed_smiles |> layer_layer_normalization()

  input_property <- layer_input(shape = c(1L), dtype = "float", name="embprop")
  embed_property <- input_property |> layer_embedding(input_dim = nrow(property_id), output_dim = 768L) |> layer_flatten()
  norm_property  <- embed_property |> layer_layer_normalization()

  concat <- list(norm_smiles,norm_property) |> layer_concatenate() |> layer_dropout(0.5)
  hidden <- concat |> layer_dense(units = 100L, activation = "sigmoid", name="emb_chemprop") |> layer_dropout(0.5)
  output <- hidden |> layer_dense(units = 1, activation="sigmoid")
  keras_model(list(embed_smiles,input_property), output)
}

mod <- build_model()
mod$compile(optimizer = optimizer_adam(), loss = "binary_crossentropy", metrics = "accuracy")
mod$fit(train, epochs = 10L, steps_per_epoch = ceiling(Ndata/100), validation_data=test)

get_activation <- function(layer,instance){
  keras_model(inputs = mod$inputs, outputs = layer$output) |> predict(instance)
}

# EVALUATION ===========================================================
embeds <- vali |> reticulate::as_iterator() |> reticulate::iter_next() |> pluck(1)

activation = get_activation(mod$layers[[9]],embeds) 
activation = tf$linalg$normalize(activation,axis=-1L)[[1]]


sim  <- tf$matmul(activation,tf$transpose(activation)) |> as.matrix()
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
