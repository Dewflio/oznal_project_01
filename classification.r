#' ---
#' title: Classification model
#' author: Ema Richnakova
#' ---

library(magrittr)
library(tidyverse)
library(dplyr) # mutate
library(caret) # confusion matrix
library(nortest) # Anderson-Darling test for normality
library(ROCit) # rocit() - optimal cut-off
library(pROC) # roc() - create ROC curve

data <- read_csv("spotify_top_songs_audio_features.csv", col_names = TRUE, num_threads = 4)
head(data)

# Counts of keys:
table(data$key)
# Counts of modes:
table(data$mode)
# Counts of time signatures:
table(data$time_signature)

#'
#'# Hypothesis 1
#' *Songs classified as major mode are more likely to have higher danceability and valence scores compared to songs classified as minor mode.*
#'

class_data <- data %>%
  mutate(mode = ifelse(mode == "Major", 1, 0)) %>%
  select_if(is.numeric)
head(class_data)

heatmap(cor(class_data),
        col = colorRampPalette(c("darkgreen", "white", "red"))(100),
        symm = TRUE)

ad.test(class_data$danceability+class_data$energy)$p.value

hist(class_data$danceability+class_data$energy, col = "green")

sample <- sample(c(TRUE, FALSE), nrow(class_data), replace = TRUE, prob = c(0.7, 0.3))
train_data <- class_data[sample, ]
test_data <- class_data[!sample, ]

model_glm <- glm(mode ~ danceability + energy, data = train_data, family = binomial)
summary(model_glm)
plot(model_glm, which = 1)

coef(model_glm)

plot(class_data$danceability + class_data$energy, class_data$mode, xlab = "Danceability + Energy", ylab = "Mode", pch = 19, col = ifelse(class_data$mode == 1, "green", "red"))
abline(model_glm, col = "black", lwd = 2)
legend("topright", legend = c("Major Mode", "Minor Mode", "Logistic Regression Line"), col = c("green", "red", "black"), pch = c(1, 1, NA), lwd = c(NA, NA, 2))

roc <- rocit(class = model_glm$y, score = model_glm$fitted.values)
par(mfrow = c(1, 2))
plot(roc)
ksplot(roc)
par(mfrow = c(1, 1))
summary(roc)

predicted_classes <- predict(model_glm, newdata = test_data, type = "response")
predicted_classes <- ifelse(predicted_classes > roc$AUC, 1, 0)

roc_curve <- roc(test_data$mode, predicted_classes)
plot(roc_curve, col = "green", main = "ROC Curve")

confusion_matrix <- caret::confusionMatrix(as_factor(predicted_classes), as_factor(test_data$mode), positive = "1")
ct <- confusion_matrix$table

TP <- ct[2, 2] # true positive
FP <- ct[2, 1] # false positive
TN <- ct[1, 1] # true negative
FN <- ct[1, 2] # false negative
P <- TP + FN # all positives (1)
N <- FP + TN # all negatives (0)

# confusion matrix heatmap
ggplot(as.data.frame(ct), aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), vjust = 1) +
  scale_fill_gradient(low = "white", high = "green") +
  labs(x = "Actual", y = "Predicted", title = "Confusion Matrix Heatmap")

print(paste("Accuracy: ", (TP + TN) / (P + N)))

print(paste("Proportion of major mode (1): ", P / (P + N)))
print(paste("Proportion of minor mode (0): ", N / (P + N)))

precision <- TP / (FP + TP)
print(paste("Precision: ", precision))

recall <- TP / P
print(paste("Recall: ", recall))

f1_score <- 2 * (precision * recall) / (precision + recall)
print(paste("F1 score: ", f1_score))

print(paste("Specificity: ", TN / (TN + FP)))

#'
#'# Hypothesis Hypothesis 2
#' *Songs with 4 beats (1) tend be less accoustic and have lower instrumentalness score.*
#' 

class_data <- data %>%
  mutate(time_signature = ifelse(time_signature == "4 beats", 1, 0)) %>%
  select_if(is.numeric)
head(class_data)

heatmap(cor(class_data),
        col = colorRampPalette(c("darkgreen", "white", "red"))(100),
        symm = TRUE)

ad.test(class_data$acousticness + class_data$instrumentalness)$p.value

hist(class_data$acousticness + class_data$instrumentalness, col = "green")

sample <- sample(c(TRUE, FALSE), nrow(class_data), replace = TRUE, prob = c(0.7, 0.3))
train_data <- class_data[sample, ]
test_data <- class_data[!sample, ]

model_glm <- glm(time_signature ~ acousticness + instrumentalness, data = train_data, family = binomial)
summary(model_glm)
plot(model_glm, which = 1)

coef(model_glm)

plot(class_data$acousticness + class_data$instrumentalness, class_data$time_signature, xlab = "Acousticness + Instrumentalness", ylab = "Time Signature", pch = 19, col = ifelse(class_data$time_signature == 1, "green", "red"))
abline(model_glm, col = "black", lwd = 2)
legend("topright", legend = c("4 beats", "other", "Logistic Regression Line"), col = c("green", "red", "black"), pch = c(1, 1, NA), lwd = c(NA, NA, 2))

roc <- rocit(class = model_glm$y, score = model_glm$fitted.values)
par(mfrow = c(1, 2))
plot(roc)
ksplot(roc)
par(mfrow = c(1, 1))
summary(roc)

predicted_classes <- predict(model_glm, newdata = test_data, type = "response")
predicted_classes <- ifelse(predicted_classes > 0.9, 1, 0)

roc_curve <- roc(test_data$time_signature, predicted_classes)
plot(roc_curve, col = "green", main = "ROC Curve")

confusion_matrix <- caret::confusionMatrix(as_factor(predicted_classes), as_factor(test_data$time_signature), positive = "1")
ct <- confusion_matrix$table

TP <- ct[2, 2] # true positive
FP <- ct[2, 1] # false positive
TN <- ct[1, 1] # true negative
FN <- ct[1, 2] # false negative
P <- TP + FN # all positives (1)
N <- FP + TN # all negatives (0)

# confusion matrix heatmap
ggplot(as.data.frame(ct), aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), vjust = 1) +
  scale_fill_gradient(low = "white", high = "green") +
  labs(x = "Actual", y = "Predicted", title = "Confusion Matrix Heatmap")

print(paste("Accuracy: ", (TP + TN) / (P + N)))

print(paste("Proportion of major mode (1): ", P / (P + N)))
print(paste("Proportion of minor mode (0): ", N / (P + N)))

precision <- TP / (FP + TP)
print(paste("Precision: ", precision))

recall <- TP / P
print(paste("Recall: ", recall))

f1_score <- 2 * (precision * recall) / (precision + recall)
print(paste("F1 score: ", f1_score))

print(paste("Specificity: ", TN / (TN + FP)))

#'
#'# FOR FUN
#' Keys correlation
#'

class_data <- data %>%
  mutate(key = ifelse(grepl("A|A#/B|B|C|C#/Db", key), 1, 0)) %>%
  select_if(is.numeric)
head(class_data)

table(class_data$key)

heatmap(cor(class_data),
        col = colorRampPalette(c("darkgreen", "white", "red"))(100),
        symm = TRUE)