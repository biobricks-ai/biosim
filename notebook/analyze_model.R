# Some scratch code for analyzing model layers and evaluating results

# LAYER ANALYSIS
raw   <- keras_model(mod$inputs[[1]],mod$get_layer("raw_embedding")$output) # 100 x 100
conv1 <- keras_model(mod$inputs[[1]],mod$get_layer("conv1d1")$output) # 100ish x 256
conv2 <- keras_model(mod$inputs[[1]],mod$get_layer("conv1d2")$output) # 100ish x 64
dense <- keras_model(mod$inputs[[1]],mod$get_layer("dense")$output) # 10


# dt <- gendt(100,eval)
# t2 <- list(arr(dt$seq1),arr(dt$seq2),arr(dt$seq3))

# output  <- dt |> mutate(output=mod$predict(t2))
# re <- raw$predict(t2[[1]])  
# c1 <- conv1$predict(t2[[1]])
# de <- dense$predict(t2[[1]])

# WHAT IS THE POSITIVE PREDICTIVE VALUE OF LOW PREDICTED DIFFERENCE
dif   <- keras_model(mod$inputs[1:2],mod$get_layer("dot_ap")$output)
dt <- eval |> select(smi1=2,seq1=seq,val1=value) |> sample_n(1000)
dt <- eval |> select(smi2=2,seq2=seq,val2=value) |> sample_n(1000) |> bind_cols(dt)
V  <- list(arr(dt$seq1),arr(dt$seq2))
dt <- dt |> mutate(de = (1-dif$predict(V)[,1])/2, equal=val1==val2)

dt |> count(equal) |> 
  pivot_wider(names_from="equal",values_from = "n") |> 
  mutate(acc=`TRUE`/(`TRUE` + `FALSE`))

dt |> filter(de>0.9) |> count(equal) |> 
  pivot_wider(names_from="equal",values_from = "n") |> 
  mutate(acc=`TRUE`/(`TRUE` + `FALSE`))

dt |> filter(de<0.0005) |> count(equal) |> 
  pivot_wider(names_from="equal",values_from = "n") |> 
  mutate(acc=`TRUE`/(`TRUE` + `FALSE`))

heatmap(dense$predict(t2[[1]]),Colv = NA,Rowv=NA)
heatmap(dense$predict(t2[[2]]),Colv = NA,Rowv=NA)
heatmap(dense$predict(t2[[3]]),Colv = NA,Rowv=NA)

# PLOT THE RESULTS
pdt  <- vdt |> 
  mutate(pred=mod$predict(vd[[1]])[,1]) |> 
  mutate(correct=ifelse(pred>0,"yes","no")) 

# ACCURACY BY GROUP
pdt |> group_by(val1) |> count(correct) |> 
  pivot_wider(id_cols="val1",names_from="correct", values_from="n") |>
  mutate(accuracy = yes / (yes+no))

# POSITIVE PREDICTIVE VALUE

pdt  <- pivot_longer(pdt,c("fpred","npred"))
ggplot(pdt,aes(dif)) + geom_histogram()
cor(pdt$value,pdt$pred)

# CAN WE OVERFIT VALIDATION SET?
hist <- fit(mod, vd[[1]], vd[[2]], batchsize=64, epochs=50, validation_split=0.1)

# SIMILARITY
similarity <- function(mols){
  
  mod  <- train_model()
  mols <- train$canonical_smiles[1:1000]

  out <- mod$predict(x[[1]])
  dot_close  <- keras_model(mod$inputs[1:2],list(mod$get_layer("dot_ac")$output), name="dot")
  
  vectors <- tokenizer$texts_to_sequences(mols) |> map(vectorize)
  vectors <- do.call(rbind,vectors)

  comb_mols    <- crossing(mols,mols) |> set_names(c("m1","m2"))
  comb_vectors <- crossing(vectors,vectors) |> set_names(c("v1","v2"))
  distances    <- dot_close$predict(list(comb_vectors$v1,comb_vectors$v2))

  molv <- train |> select(mol = canonical_smiles, value = standard_value)
  dt <- tibble(mol1 = comb_mols$m1, mol2 = comb_mols$m2, d = distances[,1]) |> arrange(-d) |> filter(mol1 != mol2)
  dt <- dt |> inner_join(molv |> rename(v1=value),by=c("mol1"="mol"))
  dt <- dt |> inner_join(molv |> rename(v2=value),by=c("mol2"="mol"))

}