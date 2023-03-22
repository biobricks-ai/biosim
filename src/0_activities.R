reticulate::use_virtualenv("./venv", required = TRUE)
pacman::p_load(biobricks, tidyverse, memoise)

activities <- bbload("chemharmony")$activities |> collect()
activities <- activities |> filter(!is.na(smiles))

# remove pid+values with less than 1000 values
activities <- activities |> group_by(pid,value) |> filter(n()>1000) |> ungroup()

# remove pids with less than 2 values
activities <- activities |> group_by(pid) |> filter(n_distinct(value)>1) |> ungroup()

# make pidnum and valnum
activities$pidnum <- activities$pid |> factor() |> as.numeric()
activities$valnum <- activities$value |> factor() |> as.numeric()

# shuffle and select
activities <- activities |> sample_frac(1.0)
activities <- activities |> select(inchi,smiles,pid,pidnum,value,valnum)

# write to the temporary cache
output <- fs::dir_create("cache/tmp") |> fs::path("activities.parquet")
activities |> arrow::write_parquet(output)