pacman::p_load(biobricks, arrow, tidyverse, magrittr)
brick_install("chemharmony")
brick_pull("chemharmony")
brick_remove("chemharmony")

activities <- brick_load("chemharmony")$activities.parquet
