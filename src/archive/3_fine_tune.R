reticulate::use_condaenv("deepchem")
pacman::p_load(biobricks, arrow, tidyverse, keras, tensorflow)

# 1. load the encoder
# 2. load the 
# 2. encode all the smiles
# 3. train an encoding tuner to predict output

m <- keras::load_model_hdf5("model/encoder.hdf5")

data <- readr::read_csv("cache/train.csv") |> 
  filter(stype=="1998110-%-Percent Effect") |>
  mutate(value = Hmisc::cut2(standard_value,g=10))

tokenizer <- text_tokenizer(31,lower=F,char_level=T,filters='') 
fit_text_tokenizer(tokenizer,data$canonical_smiles[1:1e4])

padsize   <- 100  
vectorize <- \(v){ v[1:padsize] |> modify_if(is.na,~ as.integer(0)) }
data$seq <- tokenizer$texts_to_sequences(data$canonical_smiles)
data$seq <- map(data$seq, vectorize)  

vals  <- levels(data$value)[c(1,10)]
eval  <- data |> filter(value %in% vals) |> sample_n(1000)
train <- data |> filter(value %in% vals) |> filter(!(canonical_smiles %in% eval$canonical_smiles))

gendt <- function(N,srctbl){
  s1 <- srctbl |> select(smi1=canonical_smiles,seq1=seq,val1=value) |> group_by(val1) |> sample_n(N,replace=T) |> ungroup()
  s2 <- srctbl |> select(smi2=canonical_smiles,seq2=seq,val2=value) |> group_by(val2) |> sample_n(N,replace=T) |> ungroup()
  s3 <- srctbl |> select(smi3=canonical_smiles,seq3=seq,val3=value) |> sample_n(nrow(s1),replace=T)
  bind_cols(s1,s2,s3) |> filter(val1==val2,val3 != val1) |> sample_frac(1)
}

arr   <- \(v){ do.call(rbind,v) }
gen.x <- \(dt){list(arr(dt$seq1),arr(dt$seq2),arr(dt$seq3))}
gen   <- \(N, src){ 
  \(){ 
    dt <- gendt(N,src)
    list(gen.x(dt), tf$ones(nrow(dt))) 
  }
}

source("src/models/advanced.R",local=T) # returns build_model function
mod <- build_model(padsize=100,tokenizer)
mod |> compile(optimizer = optimizer_adam(), loss=keras::loss_mean_absolute_error())    

cb <- list(callback_early_stopping("val_loss",patience=5,restore_best_weights=T, mode="min"))

vdt <- gendt(1000,eval)
vd  <- list(list(arr(vdt$seq1),arr(vdt$seq2),arr(vdt$seq3)),tf$ones(nrow(vdt)))

# Generator Fit  
x <- gen(1e6,train)()
fit(mod, x[[1]],x[[2]], batch_size=64, epochs=30, callbacks=cb, validation_data=vd)  

# WHAT IS THE POSITIVE PREDICTIVE VALUE OF LOW PREDICTED DIFFERENCE
dif   <- keras_model(mod$inputs[1:2],mod$get_layer("dot_ap")$output)
dt <- eval |> select(smi1=2,seq1=seq,val1=value) |> sample_n(1000)
dt <- eval |> select(smi2=2,seq2=seq,val2=value) |> sample_n(1000) |> bind_cols(dt)
V  <- list(arr(dt$seq1),arr(dt$seq2))
dt <- dt |> mutate(de = (1-dif$predict(V)[,1])/2, equal=val1==val2)

dt |> filter(de<0.0001) |> count(equal) |> 
  pivot_wider(names_from="equal",values_from = "n") |> 
  mutate(acc=`TRUE`/(`TRUE` + `FALSE`))
