perfect_model <- function(padsize = 100, tokenizer) {

  inp_anc <- layer_input(padsize,name = "inp-anc")  
  inp_clo <- layer_input(padsize,name = "inp-clo")
  inp_far <- layer_input(padsize,name = "inp-far")
  
  weights <- array(1:100,dim=c(100)) |> k_one_hot(num_classes=100)
  weights <- rbind(tf$zeros(100L),weights)
  one_hot_embed <- layer_embedding(name="raw_embedding", 
    input_dim=101, output_dim=100, input_length = 100, trainable=F,
    weights = list(weights))

  mp <- layer_global_max_pooling_1d(name="max_pool") 
  dn <- layer_dense(units=1,name="dense", trainable=F, use_bias=T, 
   weights=list(
    1:100 |> map_dbl(~ ifelse(.==(tokenizer$word_index$o+1),2,0)) |> array(c(100,1)),
    array(-1)
  ))

  embed <- \(x){ x |> mp() |> dn() }
  anchorv <- one_hot_embed(inp_anc) |> embed()
  positiv <- one_hot_embed(inp_clo) |> embed()
  negativ <- one_hot_embed(inp_far) |> embed()
  
  ap = (1-layer_dot(list(anchorv,positiv),axes=1,name="dot_ap",normalize=T))/2 # 0 to 1
  an = (1-layer_dot(list(anchorv,negativ),axes=1,name="dot_an",normalize=T))/2 # 0 to 1
  
  out = layer_subtract(list(an,ap),name="difference") # + true - if false
  keras_model(list(inp_anc,inp_clo,inp_far), out)
}