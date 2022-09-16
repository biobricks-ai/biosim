reticulate::use_condaenv("r-reticulate")
pacman::p_load(assertthat, glue, halfbaked, keras, tensorflow, tidyverse, abind)

reticulate::source_python("./twin.py")

defaults <- list(pad.length=100, in.dim=100, embedding.size=10, params=list(regularization=1e-8,char.embedding.size=20)) 
attach(defaults)

triplet.model = function(pad.length=100, in.dim=100, embedding.size=10,
                         params=list(regularization=1e-8,char.embedding.size=20)){
  
  input1 <- layer_input(shape = c(pad.length),name="anchor")
  input2 <- layer_input(shape = c(pad.length),name="positive")
  input3 <- layer_input(shape = c(pad.length),name="negative")
  
  embedding <- layer_embedding(
    input_dim    = in.dim, 
    output_dim   = params$char.embedding.size,
    input_length = pad.length,
    mask_zero = T,
    trainable = T,
    name = "loli-embed"
  )
  
  seq_emb <- bidirectional(
    layer=layer_lstm(units = embedding.size,activation = "tanh",recurrent_activation = "sigmoid",unroll = F,
                     use_bias = T,name ="tx-lstm"),name="lstm-embed")
  
  anchorv <- embedding(input1) |> seq_emb()
  positiv <- embedding(input2) |> seq_emb()
  negativ <- embedding(input3) |> seq_emb()
  
  distances = DistanceLayer()(anchorv,positiv,negativ)
  network   = keras_model(list(input1,input2,input3), distances)
  SiameseModel(network)
}

triplet.model.cos = function(pad.length,in.dim,embedding.size=20,
                         params=list(regularization=1e-8,char.embedding.size=20)){
  input1 <- layer_input(shape = c(pad.length),name="anchor")
  input2 <- layer_input(shape = c(pad.length),name="positive")
  input3 <- layer_input(shape = c(pad.length),name="negative")
  
  embedding <- layer_embedding(
    input_dim    = in.dim, 
    output_dim   = params$char.embedding.size,
    input_length = pad.length,
    mask_zero = T,
    trainable = T,
    embeddings_regularizer = regularizer_l2(l = params$regularization),
    name = "loli-embed"
  )
  
  seq_emb <- bidirectional(
    layer=layer_lstm(units = embedding.size,activation = "tanh",recurrent_activation = "sigmoid",unroll = F,
                     use_bias = T,name ="tx-lstm",
                     recurrent_regularizer = regularizer_l2(l = params$regularization)),name="bidir_lstm")
  
  embedding.layer = layer_dense(units=embedding.size,name="lstm-embed")
  anchorv <- embedding(input1) |> seq_emb() |> embedding.layer()
  positiv <- embedding(input2) |> seq_emb() |> embedding.layer()
  negativ <- embedding(input3) |> seq_emb() |> embedding.layer()
  
  distances = CosDistanceLayer()(anchorv,positiv,negativ)
  network   = keras_model(list(input1,input2,input3), distances)
  SiameseModel(network)
}
```

```{r}
build.vectorizer = function(data,savefile,save=T,overwrite=F){
  if(file.exists(savefile) & !overwrite){ return(readRDS(savefile)) }
  
  ctok = text_tokenizer(oov_token = "<OOV>",char_level = T,lower = F)
  ctok$fit_on_texts(glue("{data$Name}"))
  seqs = ctok$texts_to_sequences(data$Name)
  pad.length  = quantile(seqs |> sapply(length),seq(0,1,0.1))[[9]] |> ceiling()
  c.index     = tibble(char =names(ctok$word_index), index=unlist(ctok$word_index)) |> rbind(tibble(char="pad",index=0))
  
  list(
    word.index = c.index,
    pad.length = pad.length,
    text_to_seq = function(text){
      text = tolower(glue("{text}"))
      textlengths = lapply(text,\(word){1:nchar(word)})
      tibble(row = 1:length(text),char=strsplit(text,""),chri=textlengths) |> unnest(c(char,chri)) |> 
        filter(chri < pad.length) |> # Truncate
        complete(row,chri=seq(1:pad.length),fill = list(char="pad")) |> #pad
        left_join(c.index,by="char") |> replace_na(list(index=1)) |> # index `1` is the oov index
        group_by(row) |> 
        summarize(seq=array(index,dim=c(1,pad.length,1))) |> 
        ungroup() |> 
        select(seq)
    }
  )
}

twin.triplet.sample = function(in.data){
  N     = nrow(in.data)
  seqs  = in.data |> arrange(groupnum,index) |> select(anchor=seq)
  posv  = in.data |> mutate(rand = sample(1:N)) |> arrange(groupnum,rand) |> select(positive=seq)
  negv  = in.data |> slice_sample(n=N) |> select(negative=seq)
  
  bind_cols(seqs,posv,negv)
}

build.tokenized.data = function(raw.data,tokenizer,savefile,save=T,overwrite=F){
  if(file.exists(savefile) && !overwrite){
    cat("reading cache")
    readRDS(file)
  }else{
    res =  raw.data |> 
      mutate(seq = tokenizer$text_to_seq(Name)) |> unnest(seq) |>  
      group_by(CAS) |> mutate(groupcount=n()) |> filter(groupcount>=2) |> ungroup() |> 
      nest(data=!CAS) |> mutate(groupnum=row_number()) |> unnest(data) |> 
      mutate(index = row_number()) |> slice_sample(prop=1.0) |> 
      tibble()
    
    if(save){
      cat("writing cache")
      saveRDS(res,savefile)
    }
    res
  }
}

fit.triplet = function(){
  raw.data = vroom("./data/loli-synonym-list.csv") |> select(CAS,Name)
  tok      = build.vectorizer(raw.data,"./models/tok.RDS",save=T,overwrite=T) 
  assert_that(all(tok$text_to_seq("benzene")$seq == tok$text_to_seq("BENZENE")$seq))
  data     = build.tokenized.data(raw.data,tok,"./data/tokenized.data",save=T,overwrite = T)
  
  TRAIN.groups = unique(data$CAS)[1:floor(0.9*n_distinct(data$CAS))]
  TEST.groups  = setdiff(unique(data$CAS),TRAIN.groups)
  
  TRAIN.data   = data |> filter(CAS %in% TRAIN.groups)
  TEST.data    = data |> filter(CAS %in% TEST.groups)
  
  # mod  = triplet.model.cos(pad.length = tok$pad.length, in.dim = nrow(tok$word.index)+2)
  mod  = triplet.model(pad.length = tok$pad.length, in.dim = nrow(tok$word.index)+2)
  mod %>% compile(optimizer = optimizer_adam(learning_rate = 0.001))
  
  for(i in 1:20){
    
    train  = twin.triplet.sample(TRAIN.data) |> slice_sample(n = nrow(TRAIN.data))
    test   = twin.triplet.sample(TEST.data)  |> slice_sample(n = nrow(TEST.data))
    
    
    dim(train$anchor)
    dim(train$positive)
    dim(train$negative)
    
    mod |> fit(list(train$anchor,train$positive,train$negative),batch_size=128,epochs=10,
               validation_data=list(train$anchor,train$positive,train$negative))
    
    e.mod = keras_model(inputs  = mod$siamese_network$input[[1]],
                      outputs = get_layer(mod$siamese_network, "lstm-embed")$output)
    
    save_model_hdf5(e.mod,glue("./models/twin.loli.t2.{i}.hdf5"))
  }
  
   e.mod = keras_model(inputs  = mod$siamese_network$input[[1]],
                      outputs = get_layer(mod$siamese_network, "lstm-embed")$output)
  
  
  e.data = data |> mutate(e.seq=predict(e.mod,seq))
  save.embedding.model(e.mod,tok,e.data,TRAIN.groups,TEST.groups)
}

save.embedding.model = function(embedding.model,tokenizer,e.data,train.groups,test.groups){
  obj = list(tok=tokenizer,e.data=e.data,train.groups=train.groups,test.groups=test.groups)
  save_model_hdf5(embedding.model,glue("./models/twin.loli.hdf5"))
  saveRDS(obj,"./models/twin.loli.RDS")
}

save.loaded.model = function(twin.model){
  save.embedding.model(twin.model$e.mod,twin.model$tok,twin.model$e.data,twin.model$train.groups,twin.model$test.groups)
}
load.embedding.model = function(){
  e.mod   = load_model_hdf5("./models/twin.loli.hdf5")
  readRDS("./models/twin.loli.RDS") |> list_modify(e.mod=e.mod)
}
```

# Evaluate
```{r}
twin.model = load.embedding.model()
assert_that(all(twin.model$tok$text_to_seq("benzene")$seq == twin.model$tok$text_to_seq("BENZENE")$seq))

toprank = function(name,twin.model){
  v = predict(twin.model$e.mod, twin.model$tok$text_to_seq(name)[[1]])[1,]
  res = twin.model$e.data |> mutate(distance = tf$norm(tf$subtract(e.seq,v),ord="euclidean",axis=1L)$numpy())
  res |> select(index,CAS,Name,distance) |> arrange(distance)
}

toprank.instance = function(name,twin.model,top_n=20){
  toprank(name,twin.model) |> select(index,CAS,Name,distance) |> head(top_n)
}

toprank.instance("poly(styrene-co-isoprene)",twin.model,top_n = 10)
toprank.instance("polymer of isoprene and styrene",twin.model,top_n = 10)

toprank.instance("prop-2-enoic acid, methyl, polymer with benzene, ethenyl-",twin.model,top_n = 10)
toprank.instance("prop-2-enoic acid, methyl, polymer with benzene, ethenyl-",twin.model,top_n = 10)

toprank.instance("dichloroethane",twin.model,top_n = 10)
toprank.instance("benzene",       twin.model,top_n = 10)
toprank.instance("Systox",        twin.model,top_n = 10)
toprank.instance("bevacizumab",   twin.model,top_n = 10)


```

# evaluation
```{r}
twin.model = load.embedding.model()

get_topmatch = function(cas,name){
  innerfn = function(casi,namei){
    res  = toprank(namei,twin.model) |> filter(Name != namei) |> mutate(m.rank = row_number())
    res |> filter(CAS==casi) |> select(m.Name=Name,m.rank) |> slice(1:1) |> mutate(topname = res$Name[1])
  }
  pbapply::pblapply(1:length(cas),function(i){innerfn(cas[i],name[i])})
}

TEST.data  = twin.model$e.data |> filter(CAS %in% twin.model$test.groups) |> select(CAS,Name) |> 
  mutate(topmatch = get_topmatch(CAS,Name)) |> ungroup() |> unnest(topmatch)
saveRDS(TEST.data,"cosine.test.RDS")
roc.ish = TEST.data |> arrange(m.rank) |> group_by(m.rank) |> summarize(total=n()) |> ungroup() |> 
  mutate(cumsum = cumsum(total)/nrow(TEST.data))

ggplot(roc.ish,aes(x=m.rank,y=cumsum)) + geom_point()

```

# LSH
```{r}
rps = tf$random$normal(list(40L,100L),0.0,1.0,dtype=tf$dtypes$float32)$numpy()
te  = tf$random$normal(list(40L,3L),0.0,1.0,dtype=tf$dtypes$float32)

a1  = array(v1,dim=c(1,40))
sign(v1 %*% rps)
tf$multiply(v1,rps)
```

```{python}
import keras as k
DistanceLayer
custom_objects = {"DistanceLayer": DistanceLayer}
with k.utils.custom_object_scope(custom_objects):
    new_model = keras.Model.from_config(config)
k.metrics.cosine_similarity()
```
