pacman::p_load(rpart, rpart.plot, tidyverse, tidyr)

df = readr::read_csv("logs/hparam_tuning.csv") |> filter(accuracy != 0.0) |> 
  arrange(-accuracy)

# build and visualize decision tree


df <- df |> mutate(across(where(is.character),as.factor)) 

tree <- rpart(accuracy ~ num_units + prepost + combine + depth + dropout + learning_rate + activation,
  data=df)

tree <- rpart(accuracy ~ num_units + prepost + combine + depth + dropout + learning_rate + activation,
  data=df |> filter(combine=="mult", learning_rate < 0.001))


tree <- rpart(accuracy ~ num_units,
  data=df)
  
rpart.plot(tree)
# map every column in df to a factor except accuracy

library(partykit)
fdf <- df |> pivot_longer(cols = -accuracy, names_to = "variable", values_to = "value", values_transform = as.numeric)
ggplot(fdf,aes(x=value,y=accuracy)) + geom_point() + facet_wrap(~ variable, scales = "free_x") 
ctree <- ctree(accuracy ~ num_units, data = df)

