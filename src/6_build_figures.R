pacman::p_load(tidyverse)

evalsim <- readr::read_csv('cache/evalsims.csv') |> mutate(sim = sim/1000)

# create a data frame with similarities and cumulative correct above the similarity on a given row
df <- evalsim |> arrange(desc(sim)) |> mutate(cum_correct=cumsum(correct), cum_count=cumsum(N)) |> 
  mutate(cum_accuracy=cum_correct/cum_count) |>
  select(sim,cum_correct,cum_count,cum_accuracy) |> arrange(sim)

httpgd::hgd()
ggplot(df,aes(x=sim,y=cum_accuracy)) + geom_line() + 
  geom_point() + 
  theme_bw() + 
  theme(legend.position="none") + 
  labs(x='Similarity',y='Accuracy') + 
  scale_x_continuous(breaks=seq(0,1,0.1)) +
  scale_y_continuous(breaks=seq(0,1,0.1))

# What figures?
# 1. Accuracy vs. similarity
# 2. Accuracy vs. similarity, with a line for the accuracy of the best model
# 3. networx graph of the best model