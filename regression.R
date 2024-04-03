library(magrittr)
library(tidyverse)
library(dplyr) # mutate
library(caret) # confusion matrix
library(nortest) # Anderson-Darling test for normality
library(ROCit) # rocit() - optimal cut-off
library(pROC)

data <- read_csv("spotify_top_songs_audio_features.csv", col_names = TRUE, num_threads = 4)
head(data)

numeric_columns <- names(data)[sapply(data, is.numeric)]
numeric_columns
numeric_data <- data %>% select(numeric_columns)
numeric_data

numeric_correlation_matrix <- cor(pair_plot_data)
heatmap(numeric_correlation_matrix, 
        col = colorRampPalette(c("blue", "white", "red"))(100),
        symm = TRUE)

sample <- sample(c(TRUE, FALSE), nrow(data), replace = TRUE, prob = c(0.8, 0.2))
train_data <- data[sample, ]
test_data <- data[!sample, ]

hist(data$acousticness, col="green")

model <- train_data %>% glm(formula = acousticness ~ danceability + loudness + energy, family=Gamma(link="log"))
summary(model)

par(mfrow = c(1,3)) # par() sets up a plotting grid: 1 row x 3 columns
plot(model, which = c(1,2,5))



hist(data$energy, col="green")

model <- train_data %>% glm(formula = energy ~ loudness + valence, family=gaussian())
summary(model)

par(mfrow = c(1,3)) 
plot(model, which = c(1,2,5))




