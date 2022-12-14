pacman::p_load(biobricks, arrow, tidyverse, magrittr)

# CREATE ACTIVITIES TABLE ================================================
build_activities <- (function(){
  con <- DBI::dbConnect(RSQLite::SQLite(), "cache/cache.db")
  withr::defer({DBI::dbDisconnect(con)})
  DBI::dbExecute(con, "DROP TABLE IF EXISTS activities")
  DBI::dbExecute(con, "CREATE TABLE activities(activity_id, property_id, canonical_smiles, standard_value, med_prop_val, value)")
  DBI::dbExecute(con, "CREATE unique INDEX idx_activities_id ON activities(activity_id)")
  DBI::dbExecute(con, "CREATE INDEX idx_activities_smiles ON activities(canonical_smiles)")

  chembl   <- brick_load("chembl")$parquet
  assay    <- chembl$assays.parquet |> select(assay_id,assay_desc=description)
  compound <- chembl$compound_structures.parquet |> select(molregno, canonical_smiles)
  activity <- chembl$activities.parquet 

  fs.train <- activity |> inner_join(assay,by="assay_id") |> inner_join(compound,by="molregno") 

  train <- fs.train |> filter(standard_relation =="=") |> collect() |>
    mutate(stype = paste(assay_id,standard_units,standard_type,sep="-")) |>
    select(stype, canonical_smiles, standard_value)

  train <- train |> group_by(stype,canonical_smiles) |> 
    summarize(standard_value = median(standard_value)) |> 
    ungroup()

  stype.ids <- train |> group_by(stype) |> 
    summarize(property_id = uuid::UUIDgenerate()) |>  
    ungroup()
  
  activities <- train |> 
    inner_join(stype.ids,by="stype") |>
    filter(!is.na(standard_value)) |>
    filter(nchar(canonical_smiles) < 200) |>
    filter(!grepl("[+-.]",canonical_smiles)) |>
    group_by(stype) |> 
      filter(n() > 1000) |> # only keep properties with 1000+ examples
      mutate(med_prop_val=median(standard_value)) |> 
    ungroup() |>
    mutate(value = array(ifelse(standard_value>med_prop_val,1L,0L))) |>
    mutate(activity_id = row_number()) |>
    select(activity_id, property_id, canonical_smiles, standard_value, med_prop_val, value)

  DBI::dbAppendTable(con, "activities", activities)
})()

# RECORD NUMBER OF ROWS IN ACTIVITIES TABLE ===============================
con <- DBI::dbConnect(RSQLite::SQLite(), "cache/cache.db")
withr::defer({DBI::dbDisconnect(con)})
rows <- DBI::dbGetQuery(con,"SELECT count(*) FROM activities") |> pull(1)
writeLines(as.character(rows),"cache/deps/nrows-activities.dep")
