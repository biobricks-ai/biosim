reticulate::use_condaenv("r-reticulate")
pacman::p_load(biobricks, arrow, tidyverse, keras, tensorflow)

build_model <- function(padsize = 100) {
  inp_anc <- layer_input(padsize,name = "inp-anc")
  inp_clo <- layer_input(padsize,name = "inp-clo")
  inp_far <- layer_input(padsize,name = "inp-far")
  
  weights <- keras$initializers$RandomUniform(minval = minval, maxval = maxval)(array(c(99L,100L)))
  weights <- rbind(tf$zeros(100L),weights)
  smiles_embed <- layer_embedding(name="raw_embedding",
    input_dim=100, output_dim=100, input_length = padsize, mask_zero = T, weights=list(weights))
     
  smiles_lstm  <- layer_lstm(name="lstm_embedding", units=10) 

  embed_anc <- inp_anc |> smiles_embed() |> smiles_lstm()
  embed_clo <- inp_clo |> smiles_embed() |> smiles_lstm()
  embed_far <- inp_far |> smiles_embed() |> smiles_lstm()

  d1 <- layer_dot(name="dot_ac",list(embed_anc, embed_clo),axes=1L,normalize=T) # distance between anchor embedding and 'closer' embedding
  d2 <-layer_dot(name="dot_af",list(embed_anc, embed_far),axes=1L,normalize=T) # distance between anchor embedding and 'farther' embedding
  
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
  train   <- readr::read_csv("cache/train.csv") |> filter(stype=="1998110-%-Percent Effect") |> sample_n(1e4)
  
  tokenizer <- text_tokenizer(100,lower=F,char_level=T) # less than 100 unique chars in smiles
  fit_text_tokenizer(tokenizer,train$canonical_smiles[1:1e4])
  
  padsize   <- 100  
  vectorize <- \(v){ v[1:padsize] |> modify_if(is.na,~ as.integer(0)) }
  train$seq <- tokenizer$texts_to_sequences(train$canonical_smiles)
  train$seq <- map(train$seq, vectorize)  
  
  gen <- function(N=100){
    
    anc = train |> select(anc.seq=seq,anc.value=standard_value) |> sample_n(N)
    clo = train |> select(clo.seq=seq,clo.value=standard_value) |> sample_n(N)
    far = train |> select(far.seq=seq,far.value=standard_value) |> sample_n(N)
    
    # combine columns and reorder according to distance to anchor
    bind_cols(anc,clo,far) |> mutate(tmp.seq.c = clo.seq, tmp.seq.f = far.seq) |> 
      mutate(clo.seq = ifelse(abs(anc.value-clo.value) < abs(anc.value-far.value), tmp.seq.c, tmp.seq.f)) |> 
      mutate(far.seq = ifelse(abs(anc.value-clo.value) < abs(anc.value-far.value), tmp.seq.f, tmp.seq.c)) |>
      select(anc.seq,clo.seq,far.seq)
  }
  
  make_input <- \(dt, N=nrow(dt)){
    arr <- \(v){ do.call(rbind,v) }
    list(list(arr(dt$anc.seq), arr(dt$clo.seq), arr(dt$far.seq)),array(rep(1),N))
  } 
  
  mod <- build_model(padsize=100)
  mod |> compile(optimizer = optimizer_adam(), loss=loss_mean_squared_error())

  raw_emb  <- keras_model(mod$inputs[[1]],mod$get_layer("raw_embedding")$output)
  lstm_emb <- keras_model(mod$inputs[[1]],mod$get_layer("lstm_embedding")$output)
  dot_emb  <- keras_model(mod$inputs,list(mod$get_layer("dot_ac")$output,mod$get_layer("dot_af")$output))

  train_tbl <- gen(100)
  train_inp <- make_input(train_tbl)

  eg <- list(x.train[[1]][[1]][1:10,], x.train[[1]][[2]][1:10,], x.train[[1]][[3]][1:10,])

  mod$predict(eg)  
  
  t2 <- tokenizer$texts_to_sequences(c('c1ccccc1','c1ccccc1','c1cccccOH')) |> map(vectorize) 
  t2 <- do.call(rbind,t2)
  
  mod_emb  <- mod$predict(list(t2,t2,t2))
  raw_embs <- raw_emb$predict(t2)  
  dot_embs <- dot_emb$predict(list(t2,t2,t2))  
  lstm_emb <- lstm_emb$predict(t2)

  t2[1,20]
  all(embs[1,1,] == embs[2,1,])
  
  hist <- fit(mod,train_inp[[1]],train_inp[[2]], batch_size=64, epochs=200L)
}

similarity <- function(mols){
  mols = list("CCn1ccnc1C(O)c1ccccc1OC","CCS(=O)(=O)N1CCC2(CC1)c1ncc(-c3cccnc3)n1CCN2C", "Cc1cc(C(=O)N(C)Cc2ccc3c(c2)OCO3)n[nH]1")

  dot_close  <- keras_model(mod$inputs[1:2],list(mod$get_layer("dot_ac")$output), name="dot")
  vectors <- tokenizer$texts_to_sequences(mols) |> map(vectorize)
  vectors <- do.call(rbind,vectors)
  dot_close$predict(list(vectors[rep(1,3),],vectors))
}

