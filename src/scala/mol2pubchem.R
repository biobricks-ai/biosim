pacman::p_load(rJava,tidyverse)

.jinit(classpath = "fingerprinters.jar")
.jaddClassPath("fingerprinters.jar")

pubchembuilder <- function(){
  sbuilder <- .jcall("org/openscience/cdk/silent/SilentChemObjectBuilder",
                        "Lorg/openscience/cdk/interfaces/IChemObjectBuilder;",
                        "getInstance")
  
  parser = .jnew("org/openscience/cdk/smiles/SmilesParser", sbuilder)
  fp = .jnew("org/openscience/cdk/fingerprint/PubchemFingerprinter", sbuilder)

  smiles2pubchem <- function(smi){
    mol = parser$parseSmiles(smi)
    bfp = fp$getBitFingerprint(mol)
    zeros = rep(0,881)
    zeros[bfp$getSetbits()] = 1
    zeros
  }
  
  purrr::possibly(smiles2pubchem, otherwise = NA)
}
