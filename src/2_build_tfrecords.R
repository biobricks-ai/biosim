pacman::p_load(biobricks, arrow, tidyverse)

# BUILD TFRECORDS =============================================================
build_tfrecords <- function(table, chunksize){
  invisible(safely(fs::dir_delete)("./cache/tfrecord"))
  fs::dir_create("./cache/tfrecord")
  
  con <- DBI::dbConnect(RSQLite::SQLite(), "cache/cache.db")
  withr::defer({DBI::dbDisconnect(con)})

  mkquery <- function(embedding, property_id){
    DBI::dbSendQuery(con, glue::glue_sql(
      "SELECT e.canonical_smiles, e.embedding, e.arrstr, a.property_id, a.value 
      FROM embeddings e INNER JOIN activities a 
      ON e.canonical_smiles = a.canonical_smiles WHERE
      embedding = {embedding} AND property_id = {property_id}",.con=con))
  }

  write_embedding_property_value_tfrecord <- function(embedding,property_id){
    query <- mkquery(embedding, property_id)
    while( !DBI::dbHasCompleted(query) ){
      res <- DBI::dbFetch(query, n=1000) |> tibble() 
      emb <- do.call(rbind,map(res$arrstr, function(x) as.numeric(strsplit(x,",")[[1]])))
      prop <- property_id

      data <- list(
        emb  = array(res$emb,dim=c(nrow(res),length(res$emb[[1]]))),
        prop = 
        output  = array(res$value,dim=c(nrow(res)))
      )
      path <- glue::glue("./cache/tfrecord/{embedding}-{property_id}.tfrecord")
      tfrecords::write_tfrecords(data, path)
    }
  }

  prop <- tbl(con, "activities") |> select(property_id) |> distinct() |> pull(property_id)
  emb  <- tbl(con, "embeddings") |> select(embedding) |> distinct() |> pull(embedding)
  walk2(sort(rep(emb,length(prop))), rep(prop,length(emb)), write_embedding_property_value_tfrecord)
}

# what do we need?
# Pairs of compounds with similar unsupervised embeddings and their activities
# approach
# 1. get all arrays for an embedding
# 2. do a full join
# 3. get cosine similarity
# 4. filter by similarity > 0.7 (store this in pairs table)
# 5. join pairs table with activities table on left and right compound
# 6. write to tfrecord
# 7. train triplet embedding for each embedding + activity
# 8. report on triplet embedding performance relative to unsupervised embedding
#   1. 
