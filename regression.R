library(magrittr)
library(tidyverse)
library(dplyr) # mutate
library(caret) # confusion matrix
library(nortest) # Anderson-Darling test for normality
library(ROCit) # rocit() - optimal cut-off
library(pROC)

setwd("C:/Users/Leonard/Desktop/4. semester/OZNAL/Project-01/")
data <- read_csv("spotify_top_songs_audio_features.csv", col_names = TRUE, num_threads = 4)
head(data)

numeric_columns <- names(data)[sapply(data, is.numeric)]
numeric_columns
numeric_data <- data %>% select(numeric_columns)
numeric_data

numeric_correlation_matrix <- cor(numeric_data)

# Heatmap of correlations
heatmap(numeric_correlation_matrix, 
        col = colorRampPalette(c("blue", "white", "red"))(100),
        symm = TRUE)
# We can see that there is one clear feature that has a high enough (negative) correlation with a few other features
# acousticness strongly negatively correlates with loudness and energy, and somewhat strongly with danceability
# Furthermore, those three variables do not correlate much with each other (apart from the energy and loudness pair - but thats okay, we can experiment with it).
# This makes them ideal candidates for being the predictor variables for acousticness
# acousticness ~ danceability + energy + loudness

# The relationship between energy and loudness can also be explored further.
# We can see that energy also somewhat correlates wtih valence, and that valence and loudness do not correlate as much with each other.
# So we will be using this set of features for the second hypothesis.
# energy ~ loudness + valence

# Create training and testing data subsets
sample <- sample(c(TRUE, FALSE), nrow(data), replace = TRUE, prob = c(0.8, 0.2))
train_data <- data[sample, ]
test_data <- data[!sample, ]

# Find out the distribution of the acousticness variable
hist(data$acousticness, col="green")
# We can see that the feature is not normally distributed, but that it resembles a Gamma distribution - negatively skewed
# we can therefore use the general linear model, using the Gamma family. We also have to specify that link="log" for it to work with a negatively skewed distribution.
model_h1 <- train_data %>% glm(formula = acousticness ~ danceability + loudness + energy, family=Gamma(link="log"))
summary(model_h1)

par(mfrow = c(1,3)) # par() sets up a plotting grid: 1 row x 3 columns
plot(model_h1, which = c(1,2,5))



hist(data$energy, col="green")

model_h2 <- train_data %>% glm(formula = energy ~ loudness + valence, family=gaussian())
summary(model_h2)

par(mfrow = c(1,3)) 
plot(model_h2, which = c(1,2,5))

# The distribution is not normal, it is positively skewed. So we should use the Gamma family.
# But the model looks good even with gaussian
model_h2 <- train_data %>% glm(formula = energy ~ loudness + valence, family=Gamma(link="log"))
summary(model_h2)

par(mfrow = c(1,3)) 
plot(model_h2, which = c(1,2,5))




