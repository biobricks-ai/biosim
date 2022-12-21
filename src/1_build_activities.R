pacman::p_load(biobricks, arrow, tidyverse, magrittr)

# CREATE ACTIVITIES TABLE ================================================
build_activities <- (function(){
  con <- DBI::dbConnect(RSQLite::SQLite(), "cache/cache.db")
  withr::defer({DBI::dbDisconnect(con)})
  DBI::dbExecute(con, "DROP TABLE IF EXISTS activities")
  DBI::dbExecute(con, "CREATE TABLE activities(activity_id, stype, property_id, canonical_smiles, standard_value, med_prop_val, value)")
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

  activities <- train |> 
    filter(!is.na(standard_value)) |>
    filter(nchar(canonical_smiles) < 200) |>
    filter(!grepl("[+-.]",canonical_smiles)) |>
    group_by(stype) |> 
      filter(n() > 1000) |> # only keep properties with 1000+ examples
      mutate(med_prop_val=median(standard_value)) |> 
    ungroup() |>
    mutate(stype = factor(stype, levels=unique(stype))) |>
    mutate(property_id = as.numeric(stype)) |>
    mutate(value = array(ifelse(standard_value>med_prop_val,1L,0L))) |>
    mutate(activity_id = row_number()) |>
    select(activity_id, stype, property_id, canonical_smiles, standard_value, med_prop_val, value)

  DBI::dbAppendTable(con, "activities", activities)
})()

# CREATE EMBEDDINGS TABLE ================================================
build_embeddings_table <- (function(){
  con <- DBI::dbConnect(RSQLite::SQLite(), "cache/cache.db")
  DBI::dbExecute(con, "DROP TABLE IF EXISTS embeddings")
  DBI::dbExecute(con, "CREATE TABLE embeddings (canonical_smiles string, embedding string, arrstr string)")
  DBI::dbExecute(con, "CREATE UNIQUE INDEX smiles_embedding ON embeddings (canonical_smiles,embedding)")
  DBI::dbExecute(con, "CREATE INDEX idx_embeddings_smiles ON embeddings (canonical_smiles)")
  
  smiles <- DBI::dbGetQuery(con, "SELECT DISTINCT canonical_smiles FROM activities")$canonical_smiles
  DBI::dbDisconnect(con)

  source("src/scala/mol2pubchem.R", chdir = TRUE) # import mol2pubchem function
  source("src/scripts/embed_bert.R", chdir = TRUE) # import embed_bert function

  write_embedding <- function(canonical_smiles, embedding, arr){
    con <- DBI::dbConnect(RSQLite::SQLite(), "cache/cache.db")
    withr::defer({DBI::dbDisconnect(con)})
    arrstr <- apply(arr,1,\(a){ paste(as.character(a), collapse=",") })
    DBI::dbWriteTable(con, "embeddings", data.frame(canonical_smiles, embedding, arrstr), append=TRUE)
  }
  
  chunks <- halfbaked::chunk(smiles,1000)
  chunks |> iwalk(\(chunk,i){
    cat("doing chunk",i,"out of",length(chunks),"\r")
    write_embedding(chunk, "bert",    tf$math$l2_normalize(embed_bert(chunk), 1L))
    write_embedding(chunk, "pubchem", tf$math$l2_normalize(mol2pubchem(chunk),1L))
    keras::k_clear_session()
  })
})()

# WRITE EMBEDDINGS AND ACTIVITIES TABLE TO TFRECORDS ====================================
build_tfrecords <- (function(){
  pacman::p_load(tidyverse)
  invisible(safely(fs::dir_delete("./cache/tfrecord-embeddings")))
  fs::dir_create("./cache/tfrecord-embeddings")
  con <- DBI::dbConnect(RSQLite::SQLite(), "cache/cache.db")
  withr::defer({DBI::dbDisconnect(con)})

  query <- DBI::dbSendQuery(con, "SELECT canonical_smiles, embedding, arrstr FROM embeddings")
  
  i <- 1
  while( !DBI::dbHasCompleted(query) ){ 
    res <- DBI::dbFetch(query, n=1000) |> tibble() 
    emb <- do.call(rbind,map(res$arrstr, function(x) as.numeric(strsplit(x,",")[[1]])))

    data <- list(
      embedding=res$embedding, 
      arr = array(emb,dim=c(nrow(res),dim(emb)[1]))))

    path <- glue::glue("./cache/tfrecord-embeddings/tfrecord-{i}.tfrecord")
    tfrecords::write_tfrecords(data, path)
  }

})

data <- list(
  x = array(rep('a string',10),dim=c(10,1)),
  z = array(1:100,dim=c(10,10))
)
tfrecords::write_tfrecords(data, "example.tfrecords")

# CREATE PAIRS TABLE ====================================================
  
# test <- dataset |> dataset_take(5) |> coro::collect()

# build_pairs_table <- (function(){
#   con <- DBI::dbConnect(RSQLite::SQLite(), "cache/cache.db")

#   res <- DBI::dbSendQuery(con,Q)
#   while(!DBI::dbHasCompleted(res)){
#     p1 <- DBI::dbFetch(res,n=1e6) |> tibble()
#     left  <- do.call(rbind,map(p1$arra,\(a){ as.numeric(strsplit(a,",")[[1]]) }))
#     right <- do.call(rbind,map(p1$arrb,\(a){ as.numeric(strsplit(a,",")[[1]]) }))
    
#     sims <- -1*tf$losses$cosine_similarity(left,right, axis=-1L) 
#     idxs <- tf$where(sims > 0.7)[,1] |> as.array()
#     p1[idxs,]$b
#   }
# })