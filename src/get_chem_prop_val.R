reticulate::use_condaenv("r-reticulate")
pacman::p_load(biobricks, arrow, tidyverse, keras, tensorflow)

# CHEMBL_RDF ======================================================================================
# Seems too slow
from_rdf <- function(){
  chembl <- brick_load("chembl") 
  activity <- chembl$chembl_31.0_activity.parquet 

  predicates <- unique(activity$predicate)

  activity.preds <- c(
  "<http://rdf.ebi.ac.uk/terms/chembl#type>",
  "<http://rdf.ebi.ac.uk/terms/chembl#standardType>",
  "<http://rdf.ebi.ac.uk/terms/chembl#standardRelation>",
  "<http://rdf.ebi.ac.uk/terms/chembl#standardValue>",
  "<http://rdf.ebi.ac.uk/terms/chembl#standardUnits>")

  res <- activity |> filter(predicate %in% activity.preds) |> collect() 
  res2 <- res |> 
    mutate(predicate = gsub("(<http://rdf.ebi.ac.uk/terms/chembl#,>)","",predicate)) |>
    mutate(subject   = gsub("(<http://rdf.ebi.ac.uk/resource/chembl/activity/,>)","",subject)) |>
    pivot_wider(id_cols=c("subject"),names_from="predicate",values_from = "object")
}

chembl   <- brick_load("chembl")$parquet
assay    <- chembl$assays.parquet |> select(assay_id,assay_desc=description)
compound <- chembl$compound_structures.parquet |> select(molregno, canonical_smiles)
activity <- chembl$activities.parquet |> 
  select(activity_id,assay_id,molregno,
    standard_relation, standard_value,standard_units,standard_flag,standard_type) 

fs.train <- activity |> inner_join(assay,by="assay_id") |> inner_join(compound,by="molregno") 

# start easy
train <- fs.train |> filter(standard_relation =="=") |>
  collect() |>
  mutate(stype = paste(assay_id,standard_units,standard_type,sep="-")) |>
  select(stype, canonical_smiles, standard_value)
  
   
readr::write_csv(train,"/mnt/ssh/tmp/train.csv")

train <- readr::read_csv("/mnt/ssh/tmp/train.csv")
padsize <- 100
sampler <- function(){
  
  anchor  <- layer_input(100)
  closer  <- layer_input("closer", shape=c(...))
  farther <- layer_input("farther",shape=c(...))
  
  embedding <- layer_embedding(output_dim=100,...)

  embed_anchor  <- anchor  |> embedding()
  embed_closer  <- closer  |> embedding()
  embed_farther <- farther |> embedding()

  d1 <- layer_dot(list(embed_anchor, embed_closer))  # distance between anchor embedding and 'closer' embedding
  d2 <- layer_dot(list(embed_anchor, embed_farther)) # distance between anchor embedding and 'farther' embedding

  flavor  <- layer_input("assay",  shape=c(...))
  
  out <- (d2 - d1)/(d2+d1)

}
