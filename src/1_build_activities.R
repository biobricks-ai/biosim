pacman::p_load(biobricks, arrow, tidyverse)

chembl   <- brick_load("chembl")$parquet
assay    <- chembl$assays.parquet |> select(assay_id,assay_desc=description)
compound <- chembl$compound_structures.parquet |> select(molregno, canonical_smiles)
activity <- chembl$activities.parquet |> select(activity_id,assay_id,molregno,
  standard_relation, standard_value,standard_units,standard_flag,standard_type) 

fs.train <- activity |> inner_join(assay,by="assay_id") |> inner_join(compound,by="molregno") 

# start easy
train <- fs.train |> filter(standard_relation =="=") |> collect() |>
  mutate(stype = paste(assay_id,standard_units,standard_type,sep="-")) |>
  select(stype, canonical_smiles, standard_value)

cat("writing ",nrow(train),"rows\n")
readr::write_csv(train,"cache/train.csv")