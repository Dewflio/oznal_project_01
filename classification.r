#' ---
#' title: Classification model
#' author: Ema Richnakova
#' ---

library(magrittr)
library(tidyverse)
library(caret)
library(nortest) # Anderson-Darling test
library(ROCit)
library(pROC) # pROC


#' Classification - *mode*:
#'
#' * 1 - major
#' * 0 - minor

# reading data
data <- read_csv("spotify_top_songs_audio_features.csv",
                 col_names = TRUE, num_threads = 4)
# select data and hot-encode 'mode'
data %<>%
  mutate(mode = ifelse(mode == "Major", 1, 0)) %<>%
  select(artist_names, track_name, key, mode, danceability, energy, liveness,
         valence, weeks_on_chart, streams)
head(data)

#' Before every hypothesis, we should check if investigated data in hypothesis
#' are normally distributed or linear. This is crutial information before
#' choosing right classification model for our hypothesis.
#'
#' What are the differences between classification models and how to choose
#' right model?
#'
#' * Logistic Regression model
#'    + assumes that relationship between predictors and the log odds of outcome
#'      is linear (but also can handle nonlinear relationships)
#'    + may be sensitive to outliers
#' * Linear Discriminant Analysis (LDA) model
#'    + assumes that predicots are normally distributed in each class, making it
#'      robust to outliers
#'    + finds linear combination of predictors, that best separates classes

#'
#'# Hypothesis 1
#' *Songs classified as major mode are more likely to have higher danceability
#' scores compared to songs classified as minor mode.*
#' 
#' First we check, if data are normally distributed.
# Anderson-Darling test for normality
ad.test(data$danceability)$p.value
#' P-value < $\alpha = 0.05$ -> data are not normally distributed.
#' We can also check it visually on histogram. On histogram we can see, that
#' data are not normally distributed.
hist(data$danceability, col="green")
#' So we choose Logistic Regression model
#' 
#' Before creating model, we will split data as test and train sample
sample <- sample(c(TRUE, FALSE), nrow(data), replace = TRUE, prob = c(0.7, 0.3))
train_data <- data[sample, ]
test_data <- data[!sample, ]
#' Train classification model
model_glm <- glm(mode ~ danceability, data = train_data, family = binomial)
summary(model_glm)
plot(model_glm, which = 1)
#' Coefficients Interpretation:  
#' A negative coefficient suggests, that an decrease in 'danceability' is
#' associated with higher odds of the outcome (major mode).
#' The magnitude of the coefficient indicates the strength of the association.
coef(model_glm)

plot(data$danceability, data$mode, xlab = "Danceability", ylab = "Mode", pch = 19, col = ifelse(data$mode == 1, "red", "blue"))
abline(model_glm, col = "green", lwd = 2)
legend("topright", legend = c("Minor Mode", "Major Mode", "Logistic Regression Line"), col = c("blue", "red", "green"), pch = c(1, 1, NA), lwd = c(NA, NA, 2))

#' Choose optimal cut-off with ROC curve and predict test data sample.
roc <- rocit(class = model_glm$y,
score = model_glm$fitted.values)
par(mfrow = c(1,2))
plot(roc) ; ksplot(roc)
par(mfrow = c(1,1))
summary(roc)
#' Predict
predicted_classes <- predict(model_glm, newdata = test_data, type = "response")
predicted_classes <- ifelse(predicted_classes > 0.5663, 1, 0)
#'
roc_curve <- roc(test_data$mode, predicted_classes)
plot(roc_curve, col = "green", main = "ROC Curve", legacy.axes = TRUE)

# Evaluate accuracy, precision, recall, and F1-score
confusion_matrix <- caret::confusionMatrix(as_factor(predicted_classes), as_factor(test_data$mode), positive = "1")

ct <- confusion_matrix$table
# confusion matrix heatmap
ggplot(as.data.frame(ct), aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), vjust = 1) +
  scale_fill_gradient(low = "white", high = "green") +
  labs(x = "Actual", y = "Predicted", title = "Confusion Matrix Heatmap")

TP <- ct[2, 2] ; FP <- ct[2, 1]
TN <- ct[1, 1] ; FN <- ct[1, 2]
P <- TP + FN ; N <- FP + TN
(TP+TN)/(P+N)

accuracy <- (TP+TN)/(P+N)
proportion.of.1 <- P/(P+N) ; proportion.of.0 <- N/(P+N)
precision <- TP/(FP+TP)
recall <- TP/P
f1_score <- 2 * (precision * recall) / (precision + recall)
specificity <- TN / (TN+FN)
# Print evaluation metrics
print(paste("Accuracy:", accuracy)) # The proportion of correctly classified instances among all instances.
print(paste("Precision:", precision)) # The proportion of correctly predicted positive instances (major mode) among all instances predicted as positive.
print(paste("Recall:", recall)) # The proportion of correctly predicted positive instances (major mode) among all actual positive instances.
print(paste("F1-score:", f1_score)) # The harmonic mean of precision and recall, providing a balanced measure between the two.

# Interpretation of Results:
# Accuracy: Indicates overall model performance. A higher accuracy suggests better predictive performance, but it may not provide a complete picture, especially in imbalanced datasets.
# Precision: Measures the ability of the model to avoid false positives. A higher precision indicates fewer false positives, meaning fewer instances are incorrectly predicted as major mode.
# Recall: Measures the ability of the model to capture all actual positives. A higher recall indicates that the model captures a higher proportion of actual major mode instances.
# F1-score: Provides a balanced measure between precision and recall. It is useful when there is an uneven class distribution.

#'
#'# Hypothesis 2
#' *Songs in the minor mode tend to have lower liveness scores.*



#'
#'# Hypothesis 3
#' *Songs characterized by the major mode tend to have higher valence scores.*