pacman::p_load(rJava,tidyverse)

.jinit(classpath = "fingerprinters.jar")
.jaddClassPath("fingerprinters.jar")

system("pwd")
system("ls -lah")
res = system("jar -tvf fingerprinters.jar",intern=TRUE) |> keep(~ grepl("class",.))
classes = map(res, \(clz){ x=strsplit(clz," ") }) |> unlist(recursive = TRUE) |> 
  keep(~ grepl("class",.))

a = .jnew("co/insilica/vectorizer/ChemicalVectorizer")

cdk <- \(path,...){ .jnew(glue::glue("org/openscience/cdk/{path}"),...)}
sbuilder <- .jcall("org/openscience/cdk/silent/SilentChemObjectBuilder",
                        "Lorg/openscience/cdk/interfaces/IChemObjectBuilder;",
                        "getInstance")
parser = .jnew("org/openscience/cdk/smiles/SmilesParser", sbuilder)

mol = parser$parseSmiles("CC(=O)O") 

fp = .jnew("org/openscience/cdk/fingerprint/PubchemFingerprinter", sbuilder)
