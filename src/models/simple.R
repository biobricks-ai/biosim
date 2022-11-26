simple_model <- function(padsize = 100, tokenizer) {

  inp_anc <- layer_input(padsize,name = "inp-anc")  
  inp_clo <- layer_input(padsize,name = "inp-clo")
  inp_far <- layer_input(padsize,name = "inp-far")
  
  weights <- array(1:100,dim=c(100)) |> k_one_hot(num_classes=100)
  weights <- rbind(tf$zeros(100L),weights)
  one_hot_embed <- layer_embedding(name="raw_embedding", 
    input_dim=101, output_dim=100, input_length = 100, trainable=F,
    weights = list(weights))

  mp <- layer_global_max_pooling_1d(name="max_pool") 
  dn <- layer_conv_1d(filters=64,kernel_size=1,stride=1,name="conv1d")

  embed <- \(x){ x |> dn() |> mp() }
  anchorv <- one_hot_embed(inp_anc) |> embed()
  positiv <- one_hot_embed(inp_clo) |> embed()
  negativ <- one_hot_embed(inp_far) |> embed()
  
  ap = (1-layer_dot(list(anchorv,positiv),axes=1,name="dot_ap",normalize=T))/2 # 0 to 1
  an = (1-layer_dot(list(anchorv,negativ),axes=1,name="dot_an",normalize=T))/2 # 0 to 1
  
  out = layer_subtract(list(an,ap),name="difference") # + true - if false
  keras_model(list(inp_anc,inp_clo,inp_far), out)
}

sampling <- layer_lambda(f=function(arg){
  z_mean <- arg[, 1:(latent_dim)]
  z_log_var <- arg[, (latent_dim + 1):(2 * latent_dim)]
  
  epsilon <- k_random_normal(
    shape = c(k_shape(z_mean)[[1]]), 
    mean=0.,
    stddev=1.0#epsilon_std
  )
  
  z_mean + k_exp(z_log_var/2)*epsilon
})

vae_model <- function(tokenizer, original_dim=5800, latent_dim=10, intermediate_dim=100) {

  inp_anc <- layer_input(c(original_dim),name = "inp-anc")  

  h <- layer_dense(inp_anc, intermediate_dim, activation = "relu")
  z_mean <- layer_dense(h, latent_dim)
  z_log_var <- layer_dense(h, latent_dim)
  z <- layer_concatenate(list(z_mean, z_log_var)) |> sampling()
  
  # we instantiate these layers separately so as to reuse them later
  decoder_h <- layer_dense(units=intermediate_dim, activation = "relu")
  decoder_mean <- layer_dense(units = original_dim, activation = "sigmoid")
  h_decoded <- decoder_h(z)
  x_decoded_mean <- decoder_mean(h_decoded)

  # end-to-end autoencoder
  vae <- keras_model(inp_anc, x_decoded_mean)

  vae_loss <- function(x, x_decoded_mean){
    xent_loss <- (original_dim/1.0)*loss_binary_crossentropy(x, x_decoded_mean)
    kl_loss <- -0.5*k_mean(1 + z_log_var - k_square(z_mean) - k_exp(z_log_var), axis = -1L)
    xent_loss + kl_loss
  }

  vae |> compile(optimizer = "rmsprop", loss = vae_loss)

  encoder <- keras_model(inp_anc, z_mean)

  list(vae=vae,encoder=encoder)
}

