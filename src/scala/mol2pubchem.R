pacman::p_load(rJava,tidyverse)

.jinit(classpath = "fingerprinters.jar")
.jaddClassPath("fingerprinters.jar")

mol2pubchem <- function(smiles){
  sbuilder <- .jcall("org/openscience/cdk/silent/SilentChemObjectBuilder",
                        "Lorg/openscience/cdk/interfaces/IChemObjectBuilder;",
                        "getInstance")
  parser = .jnew("org/openscience/cdk/smiles/SmilesParser", sbuilder)

  l = map(smiles, \(smi){
    mol = parser$parseSmiles(smi)
    fp = .jnew("org/openscience/cdk/fingerprint/PubchemFingerprinter", sbuilder)
    bfp = fp$getBitFingerprint(mol)
    zeros = rep(0,881)
    zeros[bfp$getSetbits()] = 1
    zeros
  })
  do.call(rbind,l)
}
