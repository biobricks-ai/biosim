pacman::p_load(keras,reticulate,tensorflow,tidyverse)
reticulate::use_condaenv("deepchem")
# https://agustinus.kristia.de/techblog/2016/12/17/conditional-vae/

if(tf$executing_eagerly()){ tf$compat$v1$disable_eager_execution() }
K <- keras::backend()
source('src/models/cvae.R')

batch_size <- 100L
original_dim <- 784L
latent_dim <- 10
intermediate_dim <- 100

sampler <- function(arg){
  z_mean <- arg[, 1:(latent_dim)]
  z_log_var <- arg[, (latent_dim + 1):(2 * latent_dim)]
  epsilon <- k_random_normal(shape = c(k_shape(z_mean)[[1]]), mean=0.,stddev=1.0)
  z_mean + k_exp(z_log_var/2)*epsilon
}

vae_model <- function() {
  inp_anc  <- layer_input(c(original_dim),name = "inp-anc")  
  inp_cond <- layer_input(c(original_dim),name = "inp-cls")  
  inputs  <- layer_concatenate(list(inp_anc,inp_cond))

  sampling <- layer_lambda(f=mk_sampler(latent_dim))
  h <- layer_dense(inputs, intermediate_dim, activation = "relu")
  z_mean <- layer_dense(h, latent_dim)
  z_log_var <- layer_dense(h, latent_dim)
  z <- layer_concatenate(list(z_mean, z_log_var)) |> sampling()
  
  # we instantiate these layers separately so as to reuse them later
  decoder_h <- layer_dense(units=intermediate_dim, activation = "relu")
  decoder_mean <- layer_dense(units = original_dim, activation = "sigmoid")
  
  z_cond <- layer_concatenate(list(z,inp_cond))
  h_decoded <- decoder_h(z_cond)
  x_decoded_mean <- decoder_mean(h_decoded)

  # end-to-end autoencoder
  vae <- keras_model(list(inp_anc,inp_cond), x_decoded_mean)

  vae_loss <- function(x, x_decoded_mean){
    xent_loss <- (original_dim/1.0)*loss_binary_crossentropy(x, x_decoded_mean)
    kl_loss <- -0.5*k_mean(1 + z_log_var - k_square(z_mean) - k_exp(z_log_var), axis = -1L)
    xent_loss + kl_loss
  }

  vae |> compile(optimizer = "rmsprop", loss = vae_loss)

  encoder <- keras_model(list(inp_anc,inp_cond), z_mean)

  list(vae=vae,encoder=encoder)
}


vae = vae_model(tokenizer, original_dim = dim(X3)[[2]])
hist <- vae$vae |> fit(X3,X3,shuffle=T,epochs=100,batch_size=batch_size)
keras::save_model_hdf5(vae$encoder,"brick/encoder.hdf5")