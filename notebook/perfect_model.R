testing <- function(){
  
  data <- readr::read_csv("cache/train.csv") |> filter(stype=="1998110-%-Percent Effect") |> sample_n(1e5)
  data$standard_value  <- ifelse(grepl("o",data$canonical_smiles,ignore.case=F),1,0)
  
  tokenizer <- text_tokenizer(100,lower=F,char_level=T,filters='') # less than 100 unique chars in smiles
  fit_text_tokenizer(tokenizer,data$canonical_smiles[1:1e4])
  
  padsize   <- 100  
  vectorize <- \(v){ v[1:padsize] |> modify_if(is.na,~ as.integer(0)) }
  data$seq <- tokenizer$texts_to_sequences(data$canonical_smiles)
  data$seq <- map(data$seq, vectorize)  
  
  eval  <- sample_n(data,1000)
  train <- data |> filter(!(canonical_smiles %in% eval$canonical_smiles))

  # TESTING is c1c detectable?
  s1 <- train |> select(smi1=canonical_smiles,seq1=seq,val1=standard_value) |> filter(val1==1) |> slice(1:1000)
  s2 <- train |> select(smi2=canonical_smiles,seq2=seq,val2=standard_value) |> filter(val2==1) |> slice(1001:2000)
  s3 <- train |> select(smi3=canonical_smiles,seq3=seq,val3=standard_value) |> filter(val3==0) |> slice(2001:3000)
  dt <- bind_cols(s1,s2,s3) 

  arr <- \(v){ do.call(rbind,v) }
  x = list(list(arr(dt$seq1),arr(dt$seq2),arr(dt$seq3)),tf$ones(nrow(dt)))
  
  mod <- perfect_model(padsize=100, tokenizer)
  mod |> compile(optimizer = optimizer_adam(0.1), loss=keras::loss_mean_absolute_error())
  hist <- fit(mod, x[[1]],x[[2]], batchsize=64, epochs=50, validation_split=0.1)
  w <- mod$get_layer("dense")$get_weights()[[1]]
  w[1:tokenizer$word_index$o+1,]

  # We can train a model to fit this simple
  smod <- simple_model(padsize=100, tokenizer)
  smod |> compile(optimizer = optimizer_adam(), loss=keras::loss_mean_absolute_error())
  hist <- fit(smod, x[[1]],x[[2]], batchsize=64, epochs=50, validation_split=0.1)
}