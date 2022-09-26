build_model <- function(padsize = 100, tokenizer) {

  inp_anc <- layer_input(padsize,name = "inp-anc")  
  inp_clo <- layer_input(padsize,name = "inp-clo")
  inp_far <- layer_input(padsize,name = "inp-far")
  
  numchar <- tokenizer$num_words
  weights <- array(0:numchar,dim=c(numchar)) |> k_one_hot(num_classes=numchar)
  weights <- rbind(tf$zeros(numchar),weights)
  one_hot_embed <- layer_embedding(name="raw_embedding", 
    input_dim=numchar+1, output_dim=numchar, input_length = 100, trainable=F,
    weights = list(weights))

  mp <- layer_global_max_pooling_1d(name="max_pool") 
  conv1 <- layer_conv_1d(filters=10,kernel_size=3,stride=1,name="conv1d1",kernel_regularizer = regularizer_l1(1e-5))
  conv2 <- layer_conv_1d(filters=10,kernel_size=3,stride=1,name="conv1d2",kernel_regularizer = regularizer_l2(1e-5))
  dn <- layer_dense(units=10,name="dense",kernel_regularizer = regularizer_l2(1e-6))
  dp <- layer_dropout(rate=0.1)
  bn <- layer_batch_normalization()

  embed <- \(x){ x |> conv1() |> conv2() |> mp() |> dn() |> bn() |> dp() }
  anchorv <- one_hot_embed(inp_anc) |> embed()
  positiv <- one_hot_embed(inp_clo) |> embed()
  negativ <- one_hot_embed(inp_far) |> embed()
  
  ap = (1-layer_dot(list(anchorv,positiv),axes=1,name="dot_ap",normalize=T))/2 # 0 to 1
  an = (1-layer_dot(list(anchorv,negativ),axes=1,name="dot_an",normalize=T))/2 # 0 to 1
  
  out = layer_subtract(list(an,ap),name="difference") # + true - if false
  keras_model(list(inp_anc,inp_clo,inp_far), out)
}