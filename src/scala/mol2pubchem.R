pacman::p_load(rJava,tidyverse)

.jinit(classpath = "fingerprinters.jar")
.jaddClassPath("fingerprinters.jar")

ob <- "Lorg/openscience/cdk/interfaces/IChemObjectBuilder;"
sb <- "org/openscience/cdk/silent/SilentChemObjectBuilder"
sb <- .jcall(sb,ob,"getInstance")

parser = .jnew("org/openscience/cdk/smiles/SmilesParser", sb)
finger = .jnew("org/openscience/cdk/fingerprint/PubchemFingerprinter", sb)


s2p <- function(smi){
  mol = parser$parseSmiles(smi)
  as.list(finger$getBitFingerprint(mol)$getSetbits())
  tf$one_hot(res,881L) |> tf$reduce_sum(axis=0L)
}

embed_pubchem <- function(smi,workers=30){
  s2p <- purrr::possibly(s2p, otherwise = tf$ones(881L)*-1)
  map(smi, s2p, .progress=TRUE) |> list() |> tf$concat(axis=1L)
}

par_embed <- function(smi){
  smichunks <- halfbaked::chunk(smi, 100)
  future::plan(future::multicore(workers=30))
  
}