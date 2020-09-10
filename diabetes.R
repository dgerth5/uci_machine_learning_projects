library(caret)
library(tidyverse)
library(corrplot)

# Data :
# https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset.

# need to rename vectors, since they are all words rather than numbers
data <- data %>%
  mutate(
    gender = if_else(Gender == "Male", 1, 0),
    polyuria = if_else(Polyuria == "Yes", 1, 0),
    polydipsia = if_else(Polydipsia == "Yes", 1, 0),
    sudden_weight_loss = if_else(`sudden weight loss` == "Yes", 1, 0),
    Weakness = if_else(weakness == "Yes", 1, 0),
    polyphagia = if_else(Polyphagia == "Yes", 1, 0),
    genital_thrush = if_else(`Genital thrush` == "Yes", 1, 0),
    visual_blurring = if_else(`visual blurring` == "Yes", 1, 0),
    itching = if_else(Itching == "Yes", 1, 0),
    irritability = if_else(Irritability  == "Yes", 1, 0),
    delayed_healing = if_else(`delayed healing` == "Yes", 1, 0),
    partial_paresis = if_else(`partial paresis` == "Yes", 1, 0),
    muscle_stiffness = if_else(`muscle stiffness` == "Yes", 1, 0),
    alopecia = if_else(Alopecia == "Yes", 1, 0),
    obesity = if_else(Obesity == "Yes", 1, 0),
    Class = if_else(class == "Positive", 1, 0),
  ) %>%
  select(gender,polyuria,polydipsia,sudden_weight_loss,Weakness,polyphagia,
         genital_thrush,visual_blurring,itching,irritability,delayed_healing,
         partial_paresis,muscle_stiffness,alopecia,obesity,Class)

data$Class <- as.factor(data$Class)

# Viz
cor.matrix <- cor(data[2:14])
corrplot(cor.matrix,method = "circle")

# Data Preprocessing 
set.seed(10)
x <- floor(0.75*nrow(data))
train_index <- sample(seq_len(nrow(data)), size = x)

train <- data[train_index,]
test <- data[-train_index,]

# Modeling
tc <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
metric <- "Accuracy"

# Bayes GLM
set.seed(5)
fit.bglm <- train(Class ~., data = train, method = "bayesglm", metric = metric, trControl = tc)

# SVM
set.seed(5)
fit.svm <- train(Class ~., data = train, method="svmRadial", metric=metric, trControl=tc)

# KNN
set.seed(5)
fit.knn <- train(Class ~., data = train, method="knn", metric=metric, trControl=tc)

# Results 
check <- resamples(list(baylogi = fit.bglm, SVM = fit.svm, KNN = fit.knn))
summary(check)

# Mean Accuracy/Kappa
# SVM .945/.884, KNN .921/.839, baylog .917/.824 

# Tuning SVM
tc <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
metric <- "Accuracy"

set.seed(10)
grid <- expand.grid(.sigma = c(.01,.025,.05,.10,.15,.20), .C = seq(1, 20, by = 1))
fit.svm <- train(Class ~., data = train, method="svmRadial", metric=metric, tuneGrid = grid, trControl = tc)
print(fit.svm)
# new grid increased accuracy by close from .945 to .982, but was still increasing at sigma = .20, so i will create new grid starting at .2 to find maximum pt
# also, setting C from 1 to 20 did almost nothing, will reduce to 1 - 10
new.grid <- expand.grid(.sigma = c(.20,.25,.3,.35,.4,.45,.5), .C = seq(1, 10, by = 1))
fit.svm <- train(Class ~., data = train, method="svmRadial", metric=metric, tuneGrid = new.grid, trControl = tc)
print(fit.svm)
# sigma = 0.2 has the highest accuracy

plot(fit.svm)

# Tuning KNN
tc <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
metric <- "Accuracy"

set.seed(10)
grid <- expand.grid(.k = seq(1,20,by = 1))
fit.knn <- train(Class ~., data = train, method="knn", metric=metric, tuneGrid = grid, trControl = tc)
print(fit.knn) # Best k is k == 1, .96 accuracy

# Bagged
tc <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
metric <- "Accuracy"

# Bagged Cart
set.seed(10)
fit.bcart <- train(Class ~., data = train, method = "treebag", metric = metric, trControl = tc)

# Random Forest
set.seed(10)
fit.rf <- train(Class ~., data = train, method = "rf", metric = metric, trControl = tc)

# Boosted

# C5.0
set.seed(10)
fit.c5 <- train(Class ~., data = train, method = "C5.0", metric = metric, trControl = tc)

# SGB
set.seed(10)
fit.sgb <- train(Class ~., data = train, method = "gbm", metric = metric, trControl = tc)

new.check <- resamples(list(BAG = fit.bcart, RF = fit.rf, C50 = fit.c5, GBM = fit.sgb))
summary(new.check)

# RF/C5.0 almost the same ~ .97, GBM .964, BAG = .959

