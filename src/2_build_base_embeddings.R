pacman::p_load(biobricks, arrow, tidyverse, magrittr)

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

# LOG THE NUMBER OF ROWS IN THE EMBEDDINGS FILE FOR DVC DEPENDENCIES=====
con <- DBI::dbConnect(RSQLite::SQLite(), "cache/cache.db")
withr::defer({DBI::dbDisconnect(con)})
rows <- DBI::dbGetQuery(con,"SELECT count(*) FROM embeddings") |> pull(1)
writeLines(as.character(rows),file("cache/deps/nrows-embeddings.dep"))
