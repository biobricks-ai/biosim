pacman::p_load(biobricks,tidyverse)

activities <- brick_load("chemharmony")$harmonized$activities.parquet |> collect()

activities <- activities |> select(smiles,pid,value) |>
  group_by(smiles,pid) |> summarize(value = mean(value)) |> ungroup()

activities <- activities |> group_by(pid) |> filter(n() > 1000)

activities <- activities |> 
  filter(!is.na(smiles)) |>
  group_by(pid) |> mutate(u = mean(value), sd=sqrt(var(value)), md = median(value)) |> ungroup() |> 
  mutate(normvalue = (value - u)/sd) |>
  mutate(binvalue = ifelse(value > md,1,0)) 

activities$pidnum = activities$pid |> factor() |> as.numeric()

pids <- activities |>   
  group_by(pid) |> summarize(cnt0 = sum(binvalue==0), cnt1=sum(binvalue==1)) |> ungroup() |>  
  mutate(rat = cnt0/cnt1) |> 
  filter(rat < 1.25, rat > 0.75, cnt0 > 500, cnt1 > 500) |>
  pull(pid)

activities <- activities |> filter(pid %in% pids) |> sample_frac(1.0)

activities |>
  select(smiles,pid,pidnum,value,normvalue,binvalue) |> 
  arrow::write_parquet("cache/tmp/activities.parquet")

activities <- arrow::read_parquet("cache/tmp/activities.parquet")