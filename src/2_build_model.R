reticulate::use_condaenv("r-reticulate")
pacman::p_load(biobricks, arrow, tidyverse, keras, tensorflow)

build_model <- function(padsize = 100) {
  inp_anc <- layer_input(padsize,name = "inp-anc")
  inp_clo <- layer_input(padsize,name = "inp-clo")
  inp_far <- layer_input(padsize,name = "inp-far")
  
  smiles_embed <- layer_embedding(input_dim=100, output_dim=100, input_length = padsize) 
  smiles_lstm  <- layer_lstm(units=10) 

  embed_anc <- inp_anc |> smiles_embed() |> smiles_lstm()
  embed_clo <- inp_clo |> smiles_embed() |> smiles_lstm()
  embed_far <- inp_far |> smiles_embed() |> smiles_lstm()

  d1 <- k_square(layer_dot(list(embed_anc, embed_clo),axes=-1)) # distance between anchor embedding and 'closer' embedding
  d2 <- k_square(layer_dot(list(embed_anc, embed_far),axes=-1)) # distance between anchor embedding and 'farther' embedding
  
  # a <- layer_input(1)
  # b <- layer_input(1)
  # d <- layer_dot(list(a,b),axes=-1)
  # m <- keras_model(list(a,b),d)
  
  # m$predict(list(
  #   array(c(0,1),c(1,2)),
  #   array(c(0,1),c(1,2))))
  
  out <- ((d2 - d1) + 1)/2

  keras_model(list(inp_anc, inp_clo, inp_far), out)
}

train_model <- function(){
  train   <- readr::read_csv("datadep/train.csv") |> filter(stype=="1998110-%-Percent Effect") |> sample_n(1e4)
  
  tokenizer <- text_tokenizer(100,lower=F,char_level=T) # less than 100 unique chars in smiles
  fit_text_tokenizer(tokenizer,train$canonical_smiles[1:1e4])
  
  padsize   <- 100  
  train$seq <- tokenizer$texts_to_sequences(train$canonical_smiles)
  train$seq <- map(train$seq, \(v){ v[1:padsize] |> modify_if(is.na,~ as.integer(0)) })  
  
  gen <- function(N=100){
    
    anc = train |> select(anc.seq=seq,anc.value=standard_value) |> sample_n(N)
    clo = train |> select(clo.seq=seq,clo.value=standard_value) |> sample_n(N)
    far = train |> select(far.seq=seq,far.value=standard_value) |> sample_n(N)
    
    # combine columns and reorder according to distance to anchor
    res <- bind_cols(anc,clo,far) |> mutate(tmp.seq.c = clo.seq, tmp.seq.f = far.seq) |> 
      mutate(clo.seq = ifelse(abs(anc.value-clo.value) < abs(anc.value-far.value), tmp.seq.c, tmp.seq.f)) |> 
      mutate(far.seq = ifelse(abs(anc.value-clo.value) < abs(anc.value-far.value), tmp.seq.f, tmp.seq.c)) |>
      select(anc.seq,clo.seq,far.seq)
    
    arr <- \(v){ do.call(rbind,v) }
    list(list(arr(res$anc.seq), arr(res$clo.seq), arr(res$far.seq)),array(rep(1),N))
  } 
  
  mod <- build_model(padsize=100)
  mod |> compile(optimizer = optimizer_adam(), loss=loss_mean_squared_error())
  emb <- keras_model_sequential(mod$inputs[[1]],get_layer(mod, "lstm")$output)

  x.train <- gen(100)
  eg <- list(x.train[[1]][[1]][1:10,], x.train[[1]][[2]][1:10,], x.train[[1]][[3]][1:10,])
  mod$predict(eg)  
  
  hist <- fit(mod,x.train[[1]],x.train[[2]], batch_size=64, epochs=200L)
}

